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

package org.apache.lucene.codecs.lucene99;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.VERSION_GROUPVARINT;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.ExhaustiveVectorSearcher;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.Float16VectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.PreloadHint;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.GroupVIntUtil;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.OrdinalTranslatedKnnCollector;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.apache.lucene.util.quantization.BaseQuantizedByteVectorValues;
import org.apache.lucene.util.quantization.QuantizedVectorsReader;
import org.apache.lucene.util.quantization.ScalarQuantizer;

/**
 * Reads vectors from the index segments along with index data structures supporting KNN search.
 *
 * @lucene.experimental
 */
public final class Lucene99HnswVectorsReader extends KnnVectorsReader
    implements QuantizedVectorsReader, HnswGraphProvider {

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(Lucene99HnswVectorsFormat.class);
  // Number of ordinals to score at a time when scoring exhaustively rather than using HNSW.
  public static final int EXHAUSTIVE_BULK_SCORE_ORDS = ExhaustiveVectorSearcher.BULK_SCORE_ORDS;

  private final FlatVectorsReader flatVectorsReader;
  private final FieldInfos fieldInfos;
  private final IntObjectHashMap<FieldEntry> fields;
  private final IndexInput vectorIndex;
  private final int version;

  public Lucene99HnswVectorsReader(SegmentReadState state, FlatVectorsReader flatVectorsReader)
      throws IOException {
    this.fields = new IntObjectHashMap<>();
    this.flatVectorsReader = flatVectorsReader;
    this.fieldInfos = state.fieldInfos;
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, Lucene99HnswVectorsFormat.META_EXTENSION);
    int versionMeta = -1;
    try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
      Throwable priorE = null;
      try {
        versionMeta =
            CodecUtil.checkIndexHeader(
                meta,
                Lucene99HnswVectorsFormat.META_CODEC_NAME,
                Lucene99HnswVectorsFormat.VERSION_START,
                Lucene99HnswVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix);
        readFields(meta);
      } catch (Throwable exception) {
        priorE = exception;
      } finally {
        CodecUtil.checkFooter(meta, priorE);
      }
      this.version = versionMeta;
      this.vectorIndex =
          openDataInput(
              state,
              versionMeta,
              Lucene99HnswVectorsFormat.VECTOR_INDEX_EXTENSION,
              Lucene99HnswVectorsFormat.VECTOR_INDEX_CODEC_NAME,
              state.context.withHints(
                  // Even though this input is referred to an `indexIn`, it doesn't qualify as
                  // FileTypeHint#INDEX since it's a large file
                  FileTypeHint.DATA,
                  FileDataHint.KNN_VECTORS,
                  DataAccessHint.RANDOM,
                  PreloadHint.INSTANCE));
    } catch (Throwable t) {
      IOUtils.closeWhileSuppressingExceptions(t, this);
      throw t;
    }
  }

  private Lucene99HnswVectorsReader(
      Lucene99HnswVectorsReader reader, FlatVectorsReader flatVectorsReader) {
    this.flatVectorsReader = flatVectorsReader;
    this.fieldInfos = reader.fieldInfos;
    this.fields = reader.fields;
    this.vectorIndex = reader.vectorIndex;
    this.version = reader.version;
  }

  @Override
  public KnnVectorsReader getMergeInstance() throws IOException {
    return new Lucene99HnswVectorsReader(this, this.flatVectorsReader.getMergeInstance());
  }

  @Override
  public void finishMerge() throws IOException {
    flatVectorsReader.finishMerge();
  }

  public FlatVectorsReader getFlatVectorsReader() {
    return flatVectorsReader;
  }

  private static IndexInput openDataInput(
      SegmentReadState state,
      int versionMeta,
      String fileExtension,
      String codecName,
      IOContext context)
      throws IOException {
    String fileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, fileExtension);
    IndexInput in = state.directory.openInput(fileName, context);
    try {
      int versionVectorData =
          CodecUtil.checkIndexHeader(
              in,
              codecName,
              Lucene99HnswVectorsFormat.VERSION_START,
              Lucene99HnswVectorsFormat.VERSION_CURRENT,
              state.segmentInfo.getId(),
              state.segmentSuffix);
      if (versionMeta != versionVectorData) {
        throw new CorruptIndexException(
            "Format versions mismatch: meta="
                + versionMeta
                + ", "
                + codecName
                + "="
                + versionVectorData,
            in);
      }
      CodecUtil.retrieveChecksum(in);
      return in;
    } catch (Throwable t) {
      IOUtils.closeWhileSuppressingExceptions(t, in);
      throw t;
    }
  }

  private void readFields(ChecksumIndexInput meta) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      FieldInfo info = fieldInfos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }
      FieldEntry fieldEntry = readField(meta, info);
      validateFieldEntry(info, fieldEntry);
      fields.put(info.number, fieldEntry);
    }
  }

  private void validateFieldEntry(FieldInfo info, FieldEntry fieldEntry) {
    int dimension = info.getVectorDimension();
    if (dimension != fieldEntry.dimension) {
      throw new IllegalStateException(
          "Inconsistent vector dimension for field=\""
              + info.name
              + "\"; "
              + dimension
              + " != "
              + fieldEntry.dimension);
    }
  }

  // List of vector similarity functions. This list is defined here, in order
  // to avoid an undesirable dependency on the declaration and order of values
  // in VectorSimilarityFunction. The list values and order must be identical
  // to that of {@link o.a.l.c.l.Lucene94FieldInfosFormat#SIMILARITY_FUNCTIONS}.
  public static final List<VectorSimilarityFunction> SIMILARITY_FUNCTIONS =
      List.of(
          VectorSimilarityFunction.EUCLIDEAN,
          VectorSimilarityFunction.DOT_PRODUCT,
          VectorSimilarityFunction.COSINE,
          VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);

  public static VectorSimilarityFunction readSimilarityFunction(DataInput input)
      throws IOException {
    int i = input.readInt();
    if (i < 0 || i >= SIMILARITY_FUNCTIONS.size()) {
      throw new IllegalArgumentException("invalid distance function: " + i);
    }
    return SIMILARITY_FUNCTIONS.get(i);
  }

  public static VectorEncoding readVectorEncoding(DataInput input) throws IOException {
    int encodingId = input.readInt();
    if (encodingId < 0 || encodingId >= VectorEncoding.values().length) {
      throw new CorruptIndexException("Invalid vector encoding id: " + encodingId, input);
    }
    return VectorEncoding.values()[encodingId];
  }

  private FieldEntry readField(IndexInput input, FieldInfo info) throws IOException {
    VectorEncoding vectorEncoding = readVectorEncoding(input);
    VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
    if (similarityFunction != info.getVectorSimilarityFunction()) {
      throw new IllegalStateException(
          "Inconsistent vector similarity function for field=\""
              + info.name
              + "\"; "
              + similarityFunction
              + " != "
              + info.getVectorSimilarityFunction());
    }
    return FieldEntry.create(input, vectorEncoding, info.getVectorSimilarityFunction());
  }

  @Override
  public long ramBytesUsed() {
    return Lucene99HnswVectorsReader.SHALLOW_SIZE
        + fields.ramBytesUsed()
        + flatVectorsReader.ramBytesUsed();
  }

  @Override
  public void checkIntegrity(MergePolicy.OneMerge merge) throws IOException {
    flatVectorsReader.checkIntegrity(merge);
    CodecUtil.checksumEntireFile(vectorIndex, merge);
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    return flatVectorsReader.getFloatVectorValues(field);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    return flatVectorsReader.getByteVectorValues(field);
  }

  @Override
  public Float16VectorValues getFloat16VectorValues(String field) throws IOException {
    return flatVectorsReader.getFloat16VectorValues(field);
  }

  private FieldEntry getFieldEntryOrThrow(String field) {
    final FieldInfo info = fieldInfos.fieldInfo(field);
    final FieldEntry entry;
    if (info == null || (entry = fields.get(info.number)) == null) {
      throw new IllegalArgumentException("field=\"" + field + "\" not found");
    }
    return entry;
  }

  private FieldEntry getFieldEntry(String field, VectorEncoding expectedEncoding) {
    final FieldEntry fieldEntry = getFieldEntryOrThrow(field);
    if (fieldEntry.vectorEncoding != expectedEncoding) {
      throw new IllegalArgumentException(
          "field=\""
              + field
              + "\" is encoded as: "
              + fieldEntry.vectorEncoding
              + " expected: "
              + expectedEncoding);
    }
    return fieldEntry;
  }

  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    final FieldEntry fieldEntry = getFieldEntry(field, VectorEncoding.FLOAT32);
    search(
        fieldEntry,
        knnCollector,
        acceptDocs,
        () -> flatVectorsReader.getRandomVectorScorer(field, target));
  }

  @Override
  public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    final FieldEntry fieldEntry = getFieldEntry(field, VectorEncoding.BYTE);
    search(
        fieldEntry,
        knnCollector,
        acceptDocs,
        () -> flatVectorsReader.getRandomVectorScorer(field, target));
  }

  @Override
  public void search(String field, short[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    final FieldEntry fieldEntry = getFieldEntry(field, VectorEncoding.FLOAT16);
    search(
        fieldEntry,
        knnCollector,
        acceptDocs,
        () -> flatVectorsReader.getRandomVectorScorer(field, target));
  }

  private void search(
      FieldEntry fieldEntry,
      KnnCollector knnCollector,
      AcceptDocs acceptDocs,
      IOSupplier<RandomVectorScorer> scorerSupplier)
      throws IOException {
    if (fieldEntry.size() == 0 || knnCollector.k() == 0) {
      return;
    }
    final RandomVectorScorer scorer = scorerSupplier.get();
    final KnnCollector collector =
        new OrdinalTranslatedKnnCollector(knnCollector, scorer::ordToDoc);
    // Take into account if quantized? E.g. some scorer cost?
    // Use approximate cardinality as this is good enough, but ensure we don't exceed the graph
    // size as that is illogical
    int graphSize = (fieldEntry.vectorIndexLength() == 0) ? 0 : fieldEntry.size();
    int numVectors = scorer.maxOrd();
    assert graphSize == 0 || graphSize == numVectors;
    int filteredDocCount = Math.min(acceptDocs.cost(), graphSize);
    Bits accepted = acceptDocs.bits();
    Bits acceptedOrds = scorer.getAcceptOrds(accepted);
    // The approximate number of vectors that would be visited if we did not filter
    int unfilteredVisit = HnswGraphSearcher.expectedVisitedNodes(knnCollector.k(), graphSize);
    boolean doHnsw = knnCollector.k() < numVectors;
    if (unfilteredVisit >= filteredDocCount || graphSize == 0) {
      doHnsw = false;
    }
    if (doHnsw
        && HnswGraphSearcher.useFilteredSearch(
            knnCollector, acceptedOrds, filteredDocCount, graphSize, fieldEntry.M())
        && shouldMaterializeAcceptOrds(
            filteredDocCount, graphSize, unfilteredVisit, accepted.length())) {
      // The same bits, only answered in constant time, when the values can materialize them
      BitSet materialized = scorer.materializeAcceptOrds(accepted);
      if (materialized != null) {
        acceptedOrds = materialized;
      }
    }
    if (doHnsw) {
      HnswGraphSearcher.search(
          scorer, collector, getGraph(fieldEntry), acceptedOrds, filteredDocCount);
    } else {
      // if k is larger than the number of vectors we expect to visit in an HNSW search,
      // we can just score all the accepted vectors and collect them: exactly the accepted ordinals
      // when the values can materialize them, every ordinal tested against them otherwise
      BitSet materialized = accepted == null ? null : scorer.materializeAcceptOrds(accepted);
      ExhaustiveVectorSearcher.search(
          scorer, knnCollector, materialized != null ? materialized : acceptedOrds);
    }
  }

  /**
   * How many docs one block of the docs that have a vector covers: the blocks of {@link
   * org.apache.lucene.codecs.lucene90.IndexedDISI}, which materializing the accepted ordinals walks
   * a word at a time.
   */
  static final int DOCS_PER_BLOCK = 1 << 16;

  /**
   * How many word operations of the materialization one ordinal to doc lookup of the lazy accepted
   * ordinals costs. A lookup reads the off-heap ordinal to doc mapping through a {@code
   * DirectMonotonicReader}, which is two on-heap array reads, a bounds-checked read of the packed
   * off-heap values and a multiply-add, and then reads the accepted docs: about thirty instructions
   * with three dependent loads. A word of the materialization is two loads, an AND, a POPCNT and an
   * ADD, plus a PEXT and two shifted ORs when the word holds an accepted doc, and consecutive words
   * are independent: about four instructions with no dependent load. That puts a lookup at about
   * eight words.
   *
   * <p>This is an instruction count rather than a fitted value. It assumes that the JIT compiles
   * {@code Long#compress} to PEXT, which it does on x86 with BMI2. Where it does not, a word that
   * holds an accepted doc costs about as much as a lookup, and materializing pays off later than
   * this value says.
   */
  static final int WORD_OPS_PER_LOOKUP = 8;

  /**
   * Whether to materialize the accepted ordinals into a bit set before searching the graph with a
   * searcher optimized for filtering, rather than mapping every tested ordinal to its doc through
   * the off-heap ordinal to doc mapping.
   *
   * <p>Such a searcher tests the accepted ordinals of every neighbor of the nodes that it pops, and
   * only counts as visited the neighbors that pass and get scored, so it runs about {@code
   * unfilteredVisit * graphSize / filteredDocCount} tests, which is the estimate that {@code
   * FilteredHnswGraphSearcher} itself uses to size its visited bit set. It tests a node at most
   * once, so the graph size caps that count. The count does not depend on the number of connections
   * per node to first order: a larger fan-out makes every pop more expensive, it does not change
   * how many nodes have to be tested before enough of them pass. The estimate is rough: every
   * scored node is a test that passed, so the tests are the scored nodes divided by the fraction of
   * the docs that the filter accepts, and the searcher scores fewer nodes than {@code
   * unfilteredVisit} for a very selective filter but more for a broad one, since it also explores
   * the neighbors of the nodes that fail the filter. On a 200k doc index of random 128-dim vectors
   * with maxConn=16 and k=100, it runs 0.6 times the estimate for a filter that accepts 2% of the
   * docs and 3 to 5 times the estimate for one that accepts a fifth to a half of them.
   *
   * <p>Materializing walks the accepted docs a word at a time over the blocks of the docs that have
   * a vector, see {@code IndexedDISI#indicesOf}: {@code DOCS_PER_BLOCK / 64} words for every block
   * that holds an accepted doc, plus a bit set of {@code graphSize} bits to allocate and zero. That
   * cost depends on how many blocks the accepted docs span and not on how many there are, so it is
   * at most about {@code maxDoc / 64} words for any filter, and it saves one lookup per test. Both
   * sides are counted in word operations. The accepted docs are walked that way whether they are a
   * {@code FixedBitSet} or a {@code SparseFixedBitSet}, which is what the accept docs that Lucene
   * builds are; any other bits are leap frogged a doc at a time instead.
   *
   * @param filteredDocCount the number of docs that pass the filter
   * @param graphSize the number of nodes in the graph
   * @param unfilteredVisit the number of nodes that an unfiltered search is expected to visit
   * @param maxDoc the number of docs in the segment
   */
  static boolean shouldMaterializeAcceptOrds(
      int filteredDocCount, int graphSize, int unfilteredVisit, int maxDoc) {
    assert filteredDocCount > 0 && filteredDocCount <= graphSize && graphSize <= maxDoc;
    long tests = Math.min((long) unfilteredVisit * graphSize / filteredDocCount, graphSize);
    long blocks = Math.min(filteredDocCount, ((long) maxDoc + DOCS_PER_BLOCK - 1) / DOCS_PER_BLOCK);
    long words = blocks * (DOCS_PER_BLOCK / Long.SIZE) + graphSize / Long.SIZE;
    return tests * WORD_OPS_PER_LOOKUP >= words;
  }

  @Override
  public HnswGraph getGraph(String field) throws IOException {
    final FieldInfo info = fieldInfos.fieldInfo(field);
    final FieldEntry entry;
    if (info == null || (entry = fields.get(info.number)) == null) {
      throw new IllegalArgumentException("field=\"" + field + "\" not found");
    }
    if (entry.vectorIndexLength > 0) {
      return getGraph(entry);
    } else {
      return HnswGraph.EMPTY;
    }
  }

  private HnswGraph getGraph(FieldEntry entry) throws IOException {
    if (entry.vectorIndexLength == 0) {
      return HnswGraph.EMPTY;
    }
    return new OffHeapHnswGraph(entry, vectorIndex);
  }

  @Override
  public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
    FieldEntry entry = getFieldEntryOrThrow(fieldInfo.name);
    var flat = flatVectorsReader.getOffHeapByteSize(fieldInfo);
    var graph = Map.of(Lucene99HnswVectorsFormat.VECTOR_INDEX_EXTENSION, entry.vectorIndexLength);
    return KnnVectorsReader.mergeOffHeapByteSizeMaps(flat, graph);
  }

  @Override
  public int getVectorCount(FieldInfo fieldInfo) {
    return getFieldEntryOrThrow(fieldInfo.name).size();
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(flatVectorsReader, vectorIndex);
  }

  @Override
  public BaseQuantizedByteVectorValues getQuantizedVectorValues(String field) throws IOException {
    if (flatVectorsReader instanceof QuantizedVectorsReader qvr) {
      return qvr.getQuantizedVectorValues(field);
    }
    return null;
  }

  @Override
  public ScalarQuantizer getQuantizationState(String field) {
    if (flatVectorsReader instanceof QuantizedVectorsReader qvr) {
      return qvr.getQuantizationState(field);
    }
    return null;
  }

  @Override
  public CloseableRandomVectorScorerSupplier getRandomVectorScorerSupplierForMerge(
      FieldInfo fieldInfo, SegmentWriteState segmentWriteState) throws IOException {
    if (flatVectorsReader instanceof QuantizedVectorsReader qvr) {
      return qvr.getRandomVectorScorerSupplierForMerge(fieldInfo, segmentWriteState);
    }
    return null;
  }

  private record FieldEntry(
      VectorSimilarityFunction similarityFunction,
      VectorEncoding vectorEncoding,
      long vectorIndexOffset,
      long vectorIndexLength,
      int M,
      int numLevels,
      int dimension,
      int size,
      int[][] nodesByLevel,
      // for each level the start offsets in vectorIndex file from where to read neighbours
      DirectMonotonicReader.Meta offsetsMeta,
      long offsetsOffset,
      int offsetsBlockShift,
      long offsetsLength) {

    static FieldEntry create(
        IndexInput input,
        VectorEncoding vectorEncoding,
        VectorSimilarityFunction similarityFunction)
        throws IOException {
      final var vectorIndexOffset = input.readVLong();
      final var vectorIndexLength = input.readVLong();
      final var dimension = input.readVInt();
      final var size = input.readInt();
      // read nodes by level
      final var M = input.readVInt();
      final var numLevels = input.readVInt();
      final var nodesByLevel = new int[numLevels][];
      long numberOfOffsets = 0;
      final long offsetsOffset;
      final int offsetsBlockShift;
      final DirectMonotonicReader.Meta offsetsMeta;
      final long offsetsLength;
      for (int level = 0; level < numLevels; level++) {
        if (level > 0) {
          int numNodesOnLevel = input.readVInt();
          numberOfOffsets += numNodesOnLevel;
          nodesByLevel[level] = new int[numNodesOnLevel];
          nodesByLevel[level][0] = input.readVInt();
          for (int i = 1; i < numNodesOnLevel; i++) {
            nodesByLevel[level][i] = nodesByLevel[level][i - 1] + input.readVInt();
          }
        } else {
          numberOfOffsets += size;
        }
      }
      if (numberOfOffsets > 0) {
        offsetsOffset = input.readLong();
        offsetsBlockShift = input.readVInt();
        offsetsMeta = DirectMonotonicReader.loadMeta(input, numberOfOffsets, offsetsBlockShift);
        offsetsLength = input.readLong();
      } else {
        offsetsOffset = 0;
        offsetsBlockShift = 0;
        offsetsMeta = null;
        offsetsLength = 0;
      }
      return new FieldEntry(
          similarityFunction,
          vectorEncoding,
          vectorIndexOffset,
          vectorIndexLength,
          M,
          numLevels,
          dimension,
          size,
          nodesByLevel,
          offsetsMeta,
          offsetsOffset,
          offsetsBlockShift,
          offsetsLength);
    }
  }

  /** Read the nearest-neighbors graph from the index input */
  private final class OffHeapHnswGraph extends HnswGraph {

    final IndexInput dataIn;
    final int[][] nodesByLevel;
    final int numLevels;
    final int entryNode;
    final int size;
    int arcCount;
    int arcUpTo;
    int arc;
    private final int maxConn;
    private final DirectMonotonicReader graphLevelNodeOffsets;
    private final long[] graphLevelNodeIndexOffsets;
    // Allocated to be M*2 to track the current neighbors being explored
    private final int[] currentNeighborsBuffer;

    OffHeapHnswGraph(FieldEntry entry, IndexInput vectorIndex) throws IOException {
      this.dataIn =
          vectorIndex.slice("graph-data", entry.vectorIndexOffset, entry.vectorIndexLength);
      this.nodesByLevel = entry.nodesByLevel;
      this.numLevels = entry.numLevels;
      this.entryNode = numLevels > 1 ? nodesByLevel[numLevels - 1][0] : 0;
      this.size = entry.size();
      final RandomAccessInput addressesData =
          vectorIndex.randomAccessSlice(entry.offsetsOffset, entry.offsetsLength);
      this.graphLevelNodeOffsets =
          DirectMonotonicReader.getInstance(entry.offsetsMeta, addressesData);
      this.currentNeighborsBuffer = new int[entry.M * 2];
      this.maxConn = entry.M;
      graphLevelNodeIndexOffsets = new long[numLevels];
      graphLevelNodeIndexOffsets[0] = 0;
      for (int i = 1; i < numLevels; i++) {
        // nodesByLevel is `null` for the zeroth level as we know its size
        int nodeCount = nodesByLevel[i - 1] == null ? size : nodesByLevel[i - 1].length;
        graphLevelNodeIndexOffsets[i] = graphLevelNodeIndexOffsets[i - 1] + nodeCount;
      }
    }

    @Override
    public void seek(int level, int targetOrd) throws IOException {
      int targetIndex =
          level == 0
              ? targetOrd
              : Arrays.binarySearch(nodesByLevel[level], 0, nodesByLevel[level].length, targetOrd);
      assert targetIndex >= 0
          : "seek level=" + level + " target=" + targetOrd + " not found: " + targetIndex;
      // unsafe; no bounds checking
      dataIn.seek(graphLevelNodeOffsets.get(targetIndex + graphLevelNodeIndexOffsets[level]));
      arcCount = dataIn.readVInt();
      assert arcCount <= currentNeighborsBuffer.length : "too many neighbors: " + arcCount;
      if (arcCount > 0) {
        int sum = 0;
        if (version >= VERSION_GROUPVARINT) {
          GroupVIntUtil.readGroupVInts(dataIn, currentNeighborsBuffer, arcCount);
          for (int i = 0; i < arcCount; i++) {
            sum += currentNeighborsBuffer[i];
            currentNeighborsBuffer[i] = sum;
          }
        } else {
          for (int i = 0; i < arcCount; i++) {
            sum += dataIn.readVInt();
            currentNeighborsBuffer[i] = sum;
          }
        }
      }
      arc = -1;
      arcUpTo = 0;
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public int nextNeighbor() throws IOException {
      if (arcUpTo >= arcCount) {
        return NO_MORE_DOCS;
      }
      arc = currentNeighborsBuffer[arcUpTo];
      ++arcUpTo;
      return arc;
    }

    @Override
    public int neighborCount() {
      return arcCount;
    }

    @Override
    public int numLevels() throws IOException {
      return numLevels;
    }

    @Override
    public int maxConn() {
      return maxConn;
    }

    @Override
    public int entryNode() throws IOException {
      return entryNode;
    }

    @Override
    public NodesIterator getNodesOnLevel(int level) {
      if (level == 0) {
        return new DenseNodesIterator(size());
      } else {
        return new ArrayNodesIterator(nodesByLevel[level]);
      }
    }
  }
}
