/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.codecs.lucene104;

import com.carrotsearch.randomizedtesting.generators.RandomPicks;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.IntPredicate;
import java.util.function.LongConsumer;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter.MergeScorerData;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FilterCodecReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.FilterIndexOutput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;

/**
 * A merge that builds an HNSW graph over an asymmetric scalar encoding needs the merged vectors
 * quantized a second time, at the query bit width. {@link Lucene104ScalarQuantizedVectorsWriter}
 * produces those query-side records in the same pass that writes the merged vectors, so the merge
 * never reads them back, and hands them to {@link Lucene99HnswVectorsWriter} as a {@link
 * MergeScorerData}. These tests pin that the read-back is gone, that the graph it produces is the
 * one the read-back produced, and that the hand-off files never outlive the merge that created
 * them.
 *
 * <p>An abort landing inside the graph build, once the hand-off has been taken over by the scorer
 * supplier, is covered by {@code TestHnswMergeAbort#testRollbackDuringQuantizedMerge}, next to the
 * other abort paths.
 */
public class TestLucene104ScalarQuantizedMergeScorer extends LuceneTestCase {

  private static final int DIM = 64;

  /** Enough vectors to build a graph over, and no more: these tests all force one. */
  private static final int DOCS_PER_SEGMENT = 50;

  /**
   * Two segments of these stay under the count at which Lucene's default threshold starts paying
   * for a graph, which is what {@link #testNoHandOffWhenNoGraphIsBuilt} needs.
   */
  private static final int TINY_SEGMENT_DOCS = 200;

  private static final int MAX_CONN = 16;
  private static final int BEAM_WIDTH = 32;

  /** A threshold of 0 makes every merge build a graph, so the merge scorer is always requested. */
  private static final int ALWAYS_GRAPH = 0;

  private static final FlatVectorsFormat RAW_FORMAT =
      new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());
  private static final Lucene104ScalarQuantizedVectorScorer QUANTIZED_SCORER =
      new Lucene104ScalarQuantizedVectorScorer(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());

  private static List<ScalarEncoding> asymmetricEncodings() {
    List<ScalarEncoding> encodings = new ArrayList<>();
    for (ScalarEncoding encoding : ScalarEncoding.values()) {
      if (encoding.isAsymmetric()) {
        encodings.add(encoding);
      }
    }
    assertFalse("no asymmetric encoding to test", encodings.isEmpty());
    return encodings;
  }

  /** The shipped format: the writer prepares the merge scorer's records. */
  private static KnnVectorsFormat writerPathFormat(ScalarEncoding encoding, int threshold) {
    return new Lucene104HnswScalarQuantizedVectorsFormat(
        encoding, MAX_CONN, BEAM_WIDTH, 1, null, threshold);
  }

  /**
   * The same on-disk format, but with a flat writer that prepares nothing, so {@link
   * Lucene99HnswVectorsWriter} takes the {@code QuantizedVectorsReader} fallback that reads the
   * merged vectors back. That fallback is the behaviour this change replaces and it stays reachable
   * for every writer that does not prepare anything, so it is used here twice over: as the control
   * that makes the read-back bar a bar, and as the coverage the reader path would otherwise lose.
   */
  private static KnnVectorsFormat readerPathFormat(ScalarEncoding encoding, int threshold) {
    return new HnswOverFlatFormat(new NoPrepareFlatFormat(encoding), threshold);
  }

  /**
   * The merge must not read the vectors it just wrote. The bar is not "zero bytes": {@code
   * Lucene99HnswVectorsWriter} opens a reader over the merged segment unconditionally, and opening
   * one checks the raw vector file's index header and retrieves its checksum. What must not happen
   * is any read of the vector data itself, so the assertion is that fewer bytes are read from the
   * merged raw vector file than one single vector occupies.
   *
   * <p>The second arm is what makes the first one a gate: the same instrument, pointed at the
   * reader fallback, must see the whole merged raw vector file streamed back.
   */
  public void testMergeDoesNotReadMergedVectorsBack() throws IOException {
    for (ScalarEncoding encoding : asymmetricEncodings()) {
      for (VectorSimilarityFunction similarity :
          new VectorSimilarityFunction[] {
            VectorSimilarityFunction.EUCLIDEAN, VectorSimilarityFunction.COSINE
          }) {
        MergeCounts counts =
            runMerge(writerPathFormat(encoding, ALWAYS_GRAPH), similarity, true, true);
        assertTrue(
            "the merge read the merged vectors back: "
                + counts.mergedRawBytesRead()
                + " bytes of "
                + encoding
                + "/"
                + similarity,
            counts.mergedRawBytesRead() < (long) DIM * Float.BYTES);
      }
      MergeCounts fallback =
          runMerge(
              readerPathFormat(encoding, ALWAYS_GRAPH),
              VectorSimilarityFunction.EUCLIDEAN,
              true,
              true);
      assertTrue(
          "the reader fallback did not stream the merged vectors back: "
              + fallback.mergedRawBytesRead()
              + " bytes for "
              + encoding,
          fallback.mergedRawBytesRead() >= (long) DIM * Float.BYTES * 2 * DOCS_PER_SEGMENT);
    }
  }

  /**
   * The graph built from the writer's records is the graph the read-back built, bit for bit, and so
   * is the quantized vector data: one {@code multiScalarQuantize} call produces the same index-side
   * record as the {@code scalarQuantize} call it replaces, and the same query-side record as the
   * reader's second quantization pass.
   *
   * <p>The index is sorted and carries a second vector field that most documents do not have,
   * because the two arms write their records in whatever order the merged values come out in, and
   * nothing here may depend on that order being the dense unsorted one.
   *
   * <p>COSINE is in here because the reader side was fixed first (it quantized the query side from
   * un-normalized vectors while the index side was normalized); without that precursor the two arms
   * would differ for COSINE alone, which is why these vectors are deliberately not unit vectors.
   */
  public void testGraphIsIdenticalToTheReaderFallback() throws IOException {
    for (ScalarEncoding encoding : asymmetricEncodings()) {
      for (VectorSimilarityFunction similarity : VectorSimilarityFunction.values()) {
        assertBothPathsWriteTheSameFiles(encoding, similarity, List.of(), VECTOR_EXTENSIONS);
      }
    }
  }

  /**
   * The same, over a merge that drops deleted documents, which is where the merged ordinals stop
   * being the source ordinals.
   *
   * <p>The graph is left out of this comparison: it is not reproducible across two runs of the
   * <em>same</em> path once the merge drops documents. An A/A control that merged the reader path
   * against itself differed in {@code .vex} alone, at 3 of 3 seeds, while the same control without
   * deletions matched every file, so there is nothing here for a comparison of graphs to mean. The
   * records this change produces are compared, and they are what it writes.
   */
  public void testMergedRecordsAreIdenticalWithDeletions() throws IOException {
    for (ScalarEncoding encoding : asymmetricEncodings()) {
      VectorSimilarityFunction similarity =
          RandomPicks.randomFrom(random(), VectorSimilarityFunction.values());
      List<String> deleted = new ArrayList<>();
      for (int i = 0; i < 2 * DOCS_PER_SEGMENT; i++) {
        if (random().nextInt(10) == 0) {
          deleted.add(Integer.toString(i));
        }
      }
      assertBothPathsWriteTheSameFiles(encoding, similarity, deleted, RECORD_EXTENSIONS);
    }
  }

  private void assertBothPathsWriteTheSameFiles(
      ScalarEncoding encoding,
      VectorSimilarityFunction similarity,
      List<String> deleted,
      Set<String> compared)
      throws IOException {
    long savedSeed = HnswGraphBuilder.randSeed;
    try {
      float[][] vectors = randomVectors(2 * DOCS_PER_SEGMENT, similarity);
      long seed = random().nextLong();
      HnswGraphBuilder.randSeed = seed;
      Map<String, byte[]> writerPath =
          mergedVectorFiles(writerPathFormat(encoding, ALWAYS_GRAPH), vectors, similarity, deleted);
      HnswGraphBuilder.randSeed = seed;
      Map<String, byte[]> readerPath =
          mergedVectorFiles(readerPathFormat(encoding, ALWAYS_GRAPH), vectors, similarity, deleted);
      assertEquals(
          "different files were written for " + encoding + "/" + similarity,
          writerPath.keySet(),
          readerPath.keySet());
      for (String extension : compared) {
        assertArrayEquals(
            "the body of the merged ."
                + extension
                + " file, header and footer aside, differs for "
                + encoding
                + "/"
                + similarity,
            writerPath.get(extension),
            readerPath.get(extension));
      }
    } finally {
      HnswGraphBuilder.randSeed = savedSeed;
    }
  }

  /**
   * A merge that builds no graph asks for no merge scorer, so the writer must prepare nothing: the
   * hand-off file is not written and then deleted, it is never created. Two configurations build no
   * graph, and the writer learns about them by two different routes: a merged segment small enough
   * that a graph does not pay for itself, which {@link Lucene99HnswVectorsWriter} rules out with
   * the predicate it passes the writer, and the flat format used on its own, where nothing passes a
   * predicate at all and the two-argument merge method supplies none.
   */
  public void testNoHandOffWhenNoGraphIsBuilt() throws IOException {
    for (ScalarEncoding encoding : asymmetricEncodings()) {
      // Lucene's own default: 2 * 200 merged vectors stay under the count at which a graph starts
      // paying for itself, so the merge writes flat vectors only
      MergeCounts belowThreshold =
          runMerge(
              writerPathFormat(encoding, Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD),
              VectorSimilarityFunction.EUCLIDEAN,
              false,
              true,
              TINY_SEGMENT_DOCS);
      assertEquals(
          "a hand-off file was created for a merge that builds no graph: "
              + belowThreshold.handOffs(),
          List.of(),
          belowThreshold.handOffs());

      MergeCounts flatOnly =
          runMerge(
              new Lucene104ScalarQuantizedVectorsFormat(encoding),
              VectorSimilarityFunction.EUCLIDEAN,
              false,
              false);
      assertEquals(
          "the flat format prepared merge scorer data nobody can ask for: " + flatOnly.handOffs(),
          List.of(),
          flatOnly.handOffs());
    }
  }

  /**
   * A merge whose source readers number their fields differently from the segments behind them,
   * which is what {@link IndexWriter#addIndexes(CodecReader...)} lets a caller hand over: the merge
   * resolves the field by name all the way down to the flat writer, so a renumbered reader merges
   * like any other.
   */
  public void testMergeOfRenumberedCodecReaders() throws IOException {
    ScalarEncoding encoding = randomAsymmetricEncoding();
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(TestUtil.alwaysKnnVectorsFormat(writerPathFormat(encoding, ALWAYS_GRAPH)))
            .setUseCompoundFile(false);
    config.getCodec().compoundFormat().setShouldUseCompoundFile(false);
    try (Directory source = newDirectory();
        Directory target = newDirectory()) {
      float[][] vectors = randomVectors(2 * DOCS_PER_SEGMENT, VectorSimilarityFunction.EUCLIDEAN);
      try (IndexWriter writer = new IndexWriter(source, config)) {
        for (int i = 0; i < vectors.length; i++) {
          Document doc = new Document();
          // several more fields than the vector one, so that a renumbering can move it
          for (int f = 0; f < 5; f++) {
            doc.add(new StringField("s" + f, Integer.toString(i % 7), Field.Store.NO));
          }
          doc.add(new KnnFloatVectorField("v", vectors[i], VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(doc);
          if (i == DOCS_PER_SEGMENT - 1) {
            writer.commit();
          }
        }
        writer.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(source);
          IndexWriter writer =
              new IndexWriter(
                  target,
                  new IndexWriterConfig().setCodec(config.getCodec()).setUseCompoundFile(false))) {
        List<CodecReader> readers = new ArrayList<>();
        for (LeafReaderContext context : reader.leaves()) {
          readers.add(renumbered((CodecReader) context.reader()));
        }
        assertEquals(2, readers.size());
        writer.addIndexes(readers.toArray(new CodecReader[0]));
        writer.forceMerge(1);
      }
      assertNoTempFiles(target);
      try (DirectoryReader reader = DirectoryReader.open(target)) {
        assertEquals(vectors.length, reader.numDocs());
        KnnVectorsReader vectorsReader =
            ((CodecReader) getOnlyLeafReader(reader)).getVectorReader().unwrapReaderForField("v");
        HnswGraph graph = ((HnswGraphProvider) vectorsReader).getGraph("v");
        assertNotNull("the merged segment has no graph", graph);
        assertEquals(vectors.length, graph.size());
      }
    }
  }

  /**
   * The hand-off is a single-use resource: the graph build takes it over, and asking the same
   * handle for a second supplier is a bug the handle refuses rather than a second owner of a file
   * that is already being deleted. Closing it after it has been taken over does nothing.
   */
  public void testHandOffIsSingleUse() throws IOException {
    ScalarEncoding encoding = randomAsymmetricEncoding();
    List<MergeScorerData> handles = new ArrayList<>();
    runMerge(
        new HnswOverFlatFormat(new CapturingFlatFormat(encoding, handles), ALWAYS_GRAPH),
        VectorSimilarityFunction.EUCLIDEAN,
        true,
        true);
    assertEquals("the merge prepared no hand-off to test", 1, handles.size());
    MergeScorerData handle = handles.get(0);
    expectThrows(IllegalStateException.class, () -> handle.scorerSupplier(null));
    handle.close();
    expectThrows(IllegalStateException.class, () -> handle.scorerSupplier(null));
  }

  /**
   * A field's hand-off file is released even when its own deferred work never runs: here the second
   * field's merge fails, so the first field's records are never asked for. Nothing but closing the
   * handle can free them - by then the flat writer has not even been closed. This is also the only
   * test here that merges more than one vector field.
   */
  public void testPhaseOneFailureLeavesNoHandOff() throws IOException {
    ScalarEncoding encoding = randomAsymmetricEncoding();
    try (Directory dir =
        new FilterDirectory(newDirectory()) {
          private final AtomicBoolean firstHandOff = new AtomicBoolean(true);

          @Override
          public IndexOutput createTempOutput(String prefix, String suffix, IOContext context)
              throws IOException {
            if ("queries".equals(suffix) && firstHandOff.compareAndSet(true, false) == false) {
              throw new IOException("simulated failure creating the second hand-off file");
            }
            return super.createTempOutput(prefix, suffix, context);
          }
        }) {
      assertMergeFailsWithoutLeftovers(dir, encoding, false);
    }
  }

  /**
   * The hand-off is released when the merge fails between the pass that wrote it and the graph
   * build that was going to consume it: nothing has taken the records over, and only the handle
   * knows they exist.
   */
  public void testHandOffIsReleasedWhenItIsNeverConsumed() throws IOException {
    ScalarEncoding encoding = randomAsymmetricEncoding();
    try (Directory dir =
        new FilterDirectory(newDirectory()) {
          private final AtomicBoolean handedOff = new AtomicBoolean();

          @Override
          public IndexOutput createTempOutput(String prefix, String suffix, IOContext context)
              throws IOException {
            IndexOutput out = super.createTempOutput(prefix, suffix, context);
            if ("queries".equals(suffix)) {
              handedOff.set(true);
            }
            return out;
          }

          @Override
          public IndexInput openInput(String name, IOContext context) throws IOException {
            // the merged quantized vectors are opened when the graph build reopens the segment,
            // which is before the handle is asked for anything
            if (handedOff.get()
                && name.endsWith(
                    "." + Lucene104ScalarQuantizedVectorsFormat.VECTOR_DATA_EXTENSION)) {
              throw new IOException("simulated failure reopening the merged quantized vectors");
            }
            return super.openInput(name, context);
          }
        }) {
      assertMergeFailsWithoutLeftovers(dir, encoding, true);
    }
  }

  /** The hand-off is released when phase 2 cannot read it back either. */
  public void testPhaseTwoFailureLeavesNoHandOff() throws IOException {
    ScalarEncoding encoding = randomAsymmetricEncoding();
    try (Directory dir =
        new FilterDirectory(newDirectory()) {
          @Override
          public IndexInput openInput(String name, IOContext context) throws IOException {
            if (name.contains("queries")) {
              throw new IOException("simulated failure opening the hand-off file");
            }
            return super.openInput(name, context);
          }
        }) {
      assertMergeFailsWithoutLeftovers(dir, encoding, true);
    }
  }

  /**
   * The hand-off is released when the merge fails after the pass that wrote it: the merged vectors
   * are on disk by then, but the field's metadata still has to be written, and until the handle is
   * returned to {@link Lucene99HnswVectorsWriter} nothing else knows the file exists. That is the
   * one window no other layer covers - the flat writer's own {@code close()} deliberately releases
   * nothing.
   */
  public void testMetaFailureLeavesNoHandOff() throws IOException {
    ScalarEncoding encoding = randomAsymmetricEncoding();
    try (Directory dir =
        new FilterDirectory(newDirectory()) {
          private final AtomicBoolean handedOff = new AtomicBoolean();

          @Override
          public IndexOutput createTempOutput(String prefix, String suffix, IOContext context)
              throws IOException {
            IndexOutput out = super.createTempOutput(prefix, suffix, context);
            if ("queries".equals(suffix)) {
              // arm only once the records this test is about are being written
              handedOff.set(true);
            }
            return out;
          }

          @Override
          public IndexOutput createOutput(String name, IOContext context) throws IOException {
            IndexOutput out = super.createOutput(name, context);
            if (name.endsWith("." + Lucene104ScalarQuantizedVectorsFormat.META_EXTENSION)
                == false) {
              return out;
            }
            return new FilterIndexOutput("failing quantized meta", name, out) {
              @Override
              public void writeByte(byte b) throws IOException {
                failIfHandedOff();
                super.writeByte(b);
              }

              @Override
              public void writeBytes(byte[] b, int offset, int length) throws IOException {
                failIfHandedOff();
                super.writeBytes(b, offset, length);
              }

              private void failIfHandedOff() throws IOException {
                if (handedOff.get()) {
                  throw new IOException("simulated failure writing the quantized meta");
                }
              }
            };
          }
        }) {
      assertMergeFailsWithoutLeftovers(dir, encoding, true);
    }
  }

  /**
   * Merges two segments of one or two vector fields in a directory rigged to fail, and asserts that
   * the simulated failure is what came out and that nothing was left behind.
   */
  private void assertMergeFailsWithoutLeftovers(
      Directory dir, ScalarEncoding encoding, boolean singleField) throws IOException {
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(TestUtil.alwaysKnnVectorsFormat(writerPathFormat(encoding, ALWAYS_GRAPH)))
            .setMergeScheduler(new SerialMergeScheduler())
            .setUseCompoundFile(false);
    config.getCodec().compoundFormat().setShouldUseCompoundFile(false);
    IndexWriter writer = new IndexWriter(dir, config);
    try {
      float[][] vectors = randomVectors(2 * DOCS_PER_SEGMENT, VectorSimilarityFunction.EUCLIDEAN);
      for (int i = 0; i < vectors.length; i++) {
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("v", vectors[i], VectorSimilarityFunction.EUCLIDEAN));
        if (singleField == false) {
          doc.add(
              new KnnFloatVectorField(
                  "w", vectors[vectors.length - 1 - i], VectorSimilarityFunction.EUCLIDEAN));
        }
        writer.addDocument(doc);
        if (i == DOCS_PER_SEGMENT - 1) {
          writer.commit();
        }
      }
      writer.commit();
      Exception failure = expectThrows(Exception.class, () -> writer.forceMerge(1));
      boolean simulated = false;
      for (Throwable t = failure; t != null && simulated == false; t = t.getCause()) {
        simulated = String.valueOf(t.getMessage()).contains("simulated failure");
      }
      assertTrue("unexpected merge failure: " + failure, simulated);
      // assert before rollback: the merge itself has to have cleaned up, and IndexWriter#rollback
      // would otherwise sweep the file as an unreferenced codec file
      assertNoTempFiles(dir);
    } finally {
      // the writer may already be unusable after the failed merge, so this is best effort
      IOUtils.closeWhileHandlingException(writer::rollback);
    }
  }

  private ScalarEncoding randomAsymmetricEncoding() {
    List<ScalarEncoding> encodings = asymmetricEncodings();
    return encodings.get(random().nextInt(encodings.size()));
  }

  private static void assertNoTempFiles(Directory dir) throws IOException {
    for (String file : dir.listAll()) {
      assertFalse("a temporary file outlived the merge: " + file, file.endsWith(".tmp"));
    }
  }

  /** The same leaf, reporting a field number for every field that the segment does not use. */
  private static CodecReader renumbered(CodecReader reader) {
    List<FieldInfo> renumbered = new ArrayList<>();
    for (FieldInfo info : reader.getFieldInfos()) {
      renumbered.add(
          new FieldInfo(
              info.name,
              info.number + 100,
              info.hasTermVectors(),
              info.omitsNorms(),
              info.hasPayloads(),
              info.getIndexOptions(),
              info.getDocValuesType(),
              info.docValuesSkipIndexType(),
              info.getDocValuesGen(),
              info.attributes(),
              info.getPointDimensionCount(),
              info.getPointIndexDimensionCount(),
              info.getPointNumBytes(),
              info.getVectorDimension(),
              info.getVectorEncoding(),
              info.getVectorSimilarityFunction(),
              info.isSoftDeletesField(),
              info.isParentField()));
    }
    FieldInfos fieldInfos = new FieldInfos(renumbered.toArray(new FieldInfo[0]));
    return new FilterCodecReader(reader) {
      @Override
      public FieldInfos getFieldInfos() {
        return fieldInfos;
      }

      @Override
      public CacheHelper getCoreCacheHelper() {
        return null;
      }

      @Override
      public CacheHelper getReaderCacheHelper() {
        return null;
      }
    };
  }

  private record MergeCounts(long mergedRawBytesRead, List<String> handOffs) {}

  private MergeCounts runMerge(
      KnnVectorsFormat format,
      VectorSimilarityFunction similarity,
      boolean expectGraph,
      boolean hnsw)
      throws IOException {
    return runMerge(format, similarity, expectGraph, hnsw, DOCS_PER_SEGMENT);
  }

  /**
   * Indexes two segments, merges them, and reports what the merge read and what it created.
   *
   * @param expectGraph whether the merge is expected to build a graph
   * @param hnsw whether the format is an HNSW format at all (a flat format builds no graph and has
   *     no reader to ask about one)
   */
  private MergeCounts runMerge(
      KnnVectorsFormat format,
      VectorSimilarityFunction similarity,
      boolean expectGraph,
      boolean hnsw,
      int docsPerSegment)
      throws IOException {
    float[][] vectors = randomVectors(2 * docsPerSegment, similarity);
    try (ReadCountingDirectory dir = new ReadCountingDirectory(newDirectory())) {
      IndexWriterConfig config =
          new IndexWriterConfig()
              .setCodec(TestUtil.alwaysKnnVectorsFormat(format))
              // keep the raw vector file a file of its own rather than a region of a .cfs
              .setUseCompoundFile(false);
      config.getCodec().compoundFormat().setShouldUseCompoundFile(false);
      try (IndexWriter writer = new IndexWriter(dir, config)) {
        for (int i = 0; i < vectors.length; i++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("v", vectors[i], similarity));
          writer.addDocument(doc);
          if (i == docsPerSegment - 1) {
            writer.commit();
          }
        }
        writer.commit();
        SegmentInfos beforeMerge = SegmentInfos.readLatestCommit(dir);
        assertEquals("expected two segments to merge", 2, beforeMerge.size());
        long sourceRawBytes = 0;
        for (SegmentCommitInfo info : beforeMerge) {
          for (String file : info.files()) {
            if (file.endsWith(".vec")) {
              sourceRawBytes += dir.fileLength(file);
            }
          }
        }
        dir.resetCounts();

        writer.forceMerge(1);
        writer.commit();

        SegmentInfos afterMerge = SegmentInfos.readLatestCommit(dir);
        assertEquals("expected one segment after the merge", 1, afterMerge.size());
        SegmentCommitInfo merged = afterMerge.info(0);
        String mergedRaw = null;
        for (String file : merged.files()) {
          assertFalse("a temporary file became a segment file: " + file, file.endsWith(".tmp"));
          if (file.endsWith(".vec")) {
            mergedRaw = file;
          }
        }
        assertNotNull("the merged segment has no raw vector file", mergedRaw);
        assertNoTempFiles(dir);

        long sourceRead = 0;
        for (SegmentCommitInfo info : beforeMerge) {
          for (String file : info.files()) {
            if (file.endsWith(".vec")) {
              sourceRead += dir.bytesRead(file);
            }
          }
        }
        // positive control for the counter: the merge did read the source vectors, in full
        assertTrue(
            "the byte counter saw no read of the source vectors: " + sourceRead,
            sourceRead >= sourceRawBytes);

        if (hnsw) {
          try (DirectoryReader reader = DirectoryReader.open(writer)) {
            KnnVectorsReader vectorsReader =
                ((CodecReader) getOnlyLeafReader(reader))
                    .getVectorReader()
                    .unwrapReaderForField("v");
            HnswGraph graph = ((HnswGraphProvider) vectorsReader).getGraph("v");
            assertEquals("graph presence", expectGraph, graph != null && graph.size() > 0);
          }
        }
        return new MergeCounts(dir.bytesRead(mergedRaw), dir.handOffsCreated());
      }
    }
  }

  /**
   * Merges two segments of the given vectors and returns the bodies of the merged segment's vector
   * files, keyed by extension: the raw vectors, the quantized vectors, their metadata and the
   * graph. The body is everything between the index header and the footer, because the header
   * carries the segment's random id and the footer a checksum over it, so two indexes of the same
   * documents never agree on those bytes.
   *
   * <p>The index is sorted, a second vector field is present on a third of the documents, and the
   * documents whose ids are in {@code deleted} are deleted before the merge.
   */
  private Map<String, byte[]> mergedVectorFiles(
      KnnVectorsFormat format,
      float[][] vectors,
      VectorSimilarityFunction similarity,
      List<String> deleted)
      throws IOException {
    try (Directory dir = newDirectory()) {
      IndexWriterConfig config =
          new IndexWriterConfig()
              .setCodec(TestUtil.alwaysKnnVectorsFormat(format))
              .setIndexSort(new Sort(new SortField("sort", SortField.Type.LONG)))
              // the two arms have to meet the same segments, so nothing may merge in the background
              .setMergeScheduler(new SerialMergeScheduler())
              .setUseCompoundFile(false);
      config.getCodec().compoundFormat().setShouldUseCompoundFile(false);
      try (IndexWriter writer = new IndexWriter(dir, config)) {
        for (int i = 0; i < vectors.length; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
          doc.add(new NumericDocValuesField("sort", (i * 7919L) % 1000));
          doc.add(new KnnFloatVectorField("v", vectors[i], similarity));
          if (i % 3 == 0) {
            doc.add(new KnnFloatVectorField("w", vectors[vectors.length - 1 - i], similarity));
          }
          writer.addDocument(doc);
          if (i == vectors.length / 2 - 1) {
            writer.commit();
          }
        }
        writer.commit();
        for (String id : deleted) {
          writer.deleteDocuments(new Term("id", id));
        }
        writer.commit();
        writer.forceMerge(1);
        writer.commit();
      }
      SegmentInfos infos = SegmentInfos.readLatestCommit(dir);
      assertEquals(1, infos.size());
      Map<String, byte[]> files = new HashMap<>();
      for (String file : infos.info(0).files()) {
        String extension = file.substring(file.lastIndexOf('.') + 1);
        if (VECTOR_EXTENSIONS.contains(extension)) {
          files.put(extension, fileBody(dir, file));
        }
      }
      assertTrue("no raw vectors were written", files.containsKey("vec"));
      assertTrue("no quantized vectors were written", files.containsKey("veq"));
      assertTrue("no graph was written, so this compares nothing", files.get("vex").length > 0);
      return files;
    }
  }

  /** The records this change writes: raw vectors, quantized vectors and their metadata. */
  private static final Set<String> RECORD_EXTENSIONS = Set.of("vec", "veq", "vemq");

  private static final Set<String> VECTOR_EXTENSIONS = Set.of("vec", "veq", "vemq", "vex");

  /** The bytes of a codec file between its index header and its footer. */
  private static byte[] fileBody(Directory dir, String file) throws IOException {
    try (IndexInput in = dir.openInput(file, IOContext.READONCE)) {
      in.readInt(); // magic
      in.readString(); // codec name
      in.readInt(); // version
      in.skipBytes(StringHelper.ID_LENGTH);
      in.skipBytes(in.readByte() & 0xFF); // segment suffix
      long start = in.getFilePointer();
      long length = in.length() - CodecUtil.footerLength() - start;
      byte[] body = new byte[Math.toIntExact(length)];
      in.readBytes(body, 0, body.length);
      return body;
    }
  }

  private float[][] randomVectors(int count, VectorSimilarityFunction similarity) {
    float[][] vectors = new float[count][];
    for (int i = 0; i < count; i++) {
      float[] vector = new float[DIM];
      for (int j = 0; j < DIM; j++) {
        vector[j] = random().nextFloat() * 2 - 1;
      }
      if (similarity == VectorSimilarityFunction.DOT_PRODUCT) {
        // the only similarity that requires it; COSINE is deliberately fed non-unit vectors
        VectorUtil.l2normalize(vector);
      } else if (similarity == VectorSimilarityFunction.COSINE) {
        for (int j = 0; j < DIM; j++) {
          vector[j] *= 5f;
        }
      }
      vectors[i] = vector;
    }
    return vectors;
  }

  /** Counts every byte read from every file, through clones and slices, and names files created. */
  private static class ReadCountingDirectory extends FilterDirectory {
    private final Map<String, Long> bytesRead = new ConcurrentHashMap<>();
    private final Set<String> created = ConcurrentHashMap.newKeySet();

    ReadCountingDirectory(Directory in) {
      super(in);
    }

    @Override
    public IndexInput openInput(String name, IOContext context) throws IOException {
      return new CountingIndexInput(
          super.openInput(name, context), read -> bytesRead.merge(name, read, Long::sum));
    }

    @Override
    public IndexOutput createOutput(String name, IOContext context) throws IOException {
      created.add(name);
      return super.createOutput(name, context);
    }

    @Override
    public IndexOutput createTempOutput(String prefix, String suffix, IOContext context)
        throws IOException {
      IndexOutput output = super.createTempOutput(prefix, suffix, context);
      created.add(output.getName());
      return output;
    }

    void resetCounts() {
      bytesRead.clear();
      created.clear();
    }

    long bytesRead(String name) {
      return bytesRead.getOrDefault(name, 0L);
    }

    /** The merge scorer hand-off files created, whoever created them. */
    List<String> handOffsCreated() {
      return created.stream().filter(name -> name.contains("queries")).sorted().toList();
    }
  }

  /** Reports every byte read through it, and through everything cloned or sliced off it. */
  private static class CountingIndexInput extends FilterIndexInput {
    private final LongConsumer bytesRead;

    CountingIndexInput(IndexInput in, LongConsumer bytesRead) {
      super(in.toString(), in);
      this.bytesRead = bytesRead;
    }

    @Override
    public IndexInput clone() {
      return new CountingIndexInput(in.clone(), bytesRead);
    }

    @Override
    public IndexInput slice(String sliceDescription, long offset, long length) throws IOException {
      return new CountingIndexInput(in.slice(sliceDescription, offset, length), bytesRead);
    }

    @Override
    public byte readByte() throws IOException {
      bytesRead.accept(1);
      return in.readByte();
    }

    @Override
    public void readBytes(byte[] b, int offset, int len) throws IOException {
      bytesRead.accept(len);
      in.readBytes(b, offset, len);
    }
  }

  /**
   * An HNSW format over a flat format of the caller's choosing, writing the shipped format's bytes
   * and reusing its registered name so that the segments it writes are reopened through SPI by the
   * shipped format.
   */
  private static final class HnswOverFlatFormat extends KnnVectorsFormat {
    private static final KnnVectorsFormat READ_FORMAT =
        new Lucene104HnswScalarQuantizedVectorsFormat();

    private final FlatVectorsFormat flatVectorsFormat;
    private final int tinySegmentsThreshold;

    HnswOverFlatFormat(FlatVectorsFormat flatVectorsFormat, int tinySegmentsThreshold) {
      super(Lucene104HnswScalarQuantizedVectorsFormat.NAME);
      this.flatVectorsFormat = flatVectorsFormat;
      this.tinySegmentsThreshold = tinySegmentsThreshold;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
      return new Lucene99HnswVectorsWriter(
          state,
          MAX_CONN,
          BEAM_WIDTH,
          flatVectorsFormat,
          flatVectorsFormat.fieldsWriter(state),
          1,
          null,
          tinySegmentsThreshold);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
      return READ_FORMAT.fieldsReader(state);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
      return 1024;
    }
  }

  /** The shipped flat format with a writer that prepares nothing, as the default does. */
  private static final class NoPrepareFlatFormat extends Lucene104ScalarQuantizedVectorsFormat {
    private final ScalarEncoding encoding;

    NoPrepareFlatFormat(ScalarEncoding encoding) {
      super(encoding);
      this.encoding = encoding;
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
      return new Lucene104ScalarQuantizedVectorsWriter(
          state, encoding, RAW_FORMAT.fieldsWriter(state), QUANTIZED_SCORER) {
        @Override
        public MergeScorerData mergeOneFlatVectorFieldForMergeScorer(
            FieldInfo fieldInfo, MergeState mergeState, IntPredicate needsMergeScorer)
            throws IOException {
          mergeOneFlatVectorField(fieldInfo, mergeState);
          return null;
        }
      };
    }
  }

  /** The shipped flat format, keeping every hand-off it prepares for the test to inspect. */
  private static final class CapturingFlatFormat extends Lucene104ScalarQuantizedVectorsFormat {
    private final ScalarEncoding encoding;
    private final List<MergeScorerData> handles;

    CapturingFlatFormat(ScalarEncoding encoding, List<MergeScorerData> handles) {
      super(encoding);
      this.encoding = encoding;
      this.handles = handles;
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
      return new Lucene104ScalarQuantizedVectorsWriter(
          state, encoding, RAW_FORMAT.fieldsWriter(state), QUANTIZED_SCORER) {
        @Override
        public MergeScorerData mergeOneFlatVectorFieldForMergeScorer(
            FieldInfo fieldInfo, MergeState mergeState, IntPredicate needsMergeScorer)
            throws IOException {
          MergeScorerData handle =
              super.mergeOneFlatVectorFieldForMergeScorer(fieldInfo, mergeState, needsMergeScorer);
          if (handle != null) {
            handles.add(handle);
          }
          return handle;
        }
      };
    }
  }
}
