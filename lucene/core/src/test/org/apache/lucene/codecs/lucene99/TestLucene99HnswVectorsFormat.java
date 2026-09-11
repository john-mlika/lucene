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

import static java.lang.String.format;
import static org.apache.lucene.index.VectorSimilarityFunction.DOT_PRODUCT;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.oneOf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloat16VectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.SameThreadExecutorService;
import org.apache.lucene.util.SparseFixedBitSet;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;

public class TestLucene99HnswVectorsFormat extends BaseKnnVectorsFormatTestCase {
  @Override
  protected Codec getCodec() {
    return TestUtil.getDefaultCodec();
  }

  public void testToString() {
    FilterCodec customCodec =
        new FilterCodec("foo", Codec.getDefault()) {
          @Override
          public KnnVectorsFormat knnVectorsFormat() {
            return new Lucene99HnswVectorsFormat(10, 20);
          }
        };
    String expectedPattern =
        "Lucene99HnswVectorsFormat(name=Lucene99HnswVectorsFormat, maxConn=10, beamWidth=20, tinySegmentsThreshold=100, flatVectorFormat=Lucene99FlatVectorsFormat(vectorsScorer=%s()))";
    var defaultScorer = format(Locale.ROOT, expectedPattern, "DefaultFlatVectorScorer");
    var memSegScorer =
        format(Locale.ROOT, expectedPattern, "Lucene99MemorySegmentFlatVectorsScorer");
    assertThat(customCodec.knnVectorsFormat().toString(), is(oneOf(defaultScorer, memSegScorer)));
  }

  public void testLimits() {
    expectThrows(IllegalArgumentException.class, () -> new Lucene99HnswVectorsFormat(-1, 20));
    expectThrows(IllegalArgumentException.class, () -> new Lucene99HnswVectorsFormat(0, 20));
    expectThrows(IllegalArgumentException.class, () -> new Lucene99HnswVectorsFormat(20, 0));
    expectThrows(IllegalArgumentException.class, () -> new Lucene99HnswVectorsFormat(20, -1));
    expectThrows(IllegalArgumentException.class, () -> new Lucene99HnswVectorsFormat(512 + 1, 20));
    expectThrows(IllegalArgumentException.class, () -> new Lucene99HnswVectorsFormat(20, 3201));
    expectThrows(
        IllegalArgumentException.class,
        () -> new Lucene99HnswVectorsFormat(20, 100, 1, new SameThreadExecutorService()));
  }

  public void testSimpleOffHeapSize() throws IOException {
    float[] vector = randomVector(random().nextInt(12, 500));
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
      Document doc = new Document();
      doc.add(new KnnFloatVectorField("f", vector, DOT_PRODUCT));
      w.addDocument(doc);
      w.commit();
      try (IndexReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);
        if (r instanceof CodecReader codecReader) {
          KnnVectorsReader knnVectorsReader = codecReader.getVectorReader();
          knnVectorsReader = knnVectorsReader.unwrapReaderForField("f");
          var fieldInfo = r.getFieldInfos().fieldInfo("f");
          var offHeap = knnVectorsReader.getOffHeapByteSize(fieldInfo);
          assertEquals(vector.length * Float.BYTES, (long) offHeap.get("vec"));
          assertNotNull(offHeap.get("vex"));
          assertEquals(2, offHeap.size());
        }
      }
    }
  }

  @Override
  protected boolean supportsFloatVectorFallback() {
    return false;
  }

  /**
   * A filtered graph search of a field where some docs have no vector materializes the accepted
   * ordinals, and must return exactly what the same search of a field where every doc has one
   * returns.
   */
  public void testFilteredSearchOfFieldWithDocsWithoutVectors() throws IOException {
    int numDocs = 6000;
    int dimension = 8;
    int k = 10;
    int maxAcceptedDoc = 749;
    float[][] vectors = new float[numDocs][];
    int[] ordOfDoc = new int[numDocs];
    int numVectors = 0;
    for (int i = 0; i < numDocs; i++) {
      // Every tenth doc has no vector, so that vector ordinals and doc ids differ
      ordOfDoc[i] = -1;
      if (i % 10 != 3) {
        vectors[i] = randomVector(dimension);
        ordOfDoc[i] = numVectors++;
      }
    }
    float[] queryVector = randomVector(dimension);

    try (Directory sparseDir = newDirectory();
        Directory denseDir = newDirectory()) {
      // The same vectors in the same order in both indexes, so that they get the same graph. Only
      // the sparse index also holds the docs that have no vector.
      indexVectors(sparseDir, vectors, true);
      indexVectors(denseDir, vectors, false);

      try (DirectoryReader sparseReader = DirectoryReader.open(sparseDir);
          DirectoryReader denseReader = DirectoryReader.open(denseDir)) {
        assertEquals(1, sparseReader.leaves().size());
        assertEquals(1, denseReader.leaves().size());
        LeafReader sparseLeaf = sparseReader.leaves().get(0).reader();
        LeafReader denseLeaf = denseReader.leaves().get(0).reader();
        assertEquals(numDocs, sparseLeaf.maxDoc());
        assertEquals(numVectors, denseLeaf.maxDoc());

        // The docs below maxAcceptedDoc that have a vector, and the same vectors in the dense
        // index. Both searches then see the same accepted ordinals and the same filter size, so
        // they must return the same hits.
        FixedBitSet sparseAccepted = new FixedBitSet(numDocs);
        FixedBitSet denseAccepted = new FixedBitSet(numVectors);
        for (int doc = 0; doc <= maxAcceptedDoc; doc++) {
          if (vectors[doc] != null) {
            sparseAccepted.set(doc);
            denseAccepted.set(ordOfDoc[doc]);
          }
        }
        int accepted = sparseAccepted.cardinality();
        assertEquals(accepted, denseAccepted.cardinality());
        assertGraphSearchMaterializesAcceptedOrds(accepted, numVectors, k, numDocs);

        // The visited limit is unlimited, so nothing here can fall back to an exact search
        TopDocs sparseDocs =
            sparseLeaf.searchNearestVectors(
                "field",
                queryVector,
                k,
                AcceptDocs.fromLiveDocs(sparseAccepted, numDocs),
                Integer.MAX_VALUE);
        TopDocs denseDocs =
            denseLeaf.searchNearestVectors(
                "field",
                queryVector,
                k,
                AcceptDocs.fromLiveDocs(denseAccepted, numVectors),
                Integer.MAX_VALUE);
        assertEquals(k, sparseDocs.scoreDocs.length);
        assertEquals(k, denseDocs.scoreDocs.length);
        for (int i = 0; i < k; i++) {
          int doc = sparseDocs.scoreDocs[i].doc;
          assertTrue(sparseAccepted.get(doc));
          assertEquals(ordOfDoc[doc], denseDocs.scoreDocs[i].doc);
          assertEquals(denseDocs.scoreDocs[i].score, sparseDocs.scoreDocs[i].score, 0f);
        }

        // The filter now also accepts the docs that have no vector: they take part in the leap
        // frog and in the filter size, and none of them can be a hit
        FixedBitSet acceptedWithHoles = new FixedBitSet(numDocs);
        acceptedWithHoles.set(0, maxAcceptedDoc + 1);
        assertGraphSearchMaterializesAcceptedOrds(maxAcceptedDoc + 1, numVectors, k, numDocs);
        TopDocs sparseDocsWithHoles =
            sparseLeaf.searchNearestVectors(
                "field",
                queryVector,
                k,
                AcceptDocs.fromLiveDocs(acceptedWithHoles, numDocs),
                Integer.MAX_VALUE);
        assertEquals(k, sparseDocsWithHoles.scoreDocs.length);
        for (ScoreDoc hit : sparseDocsWithHoles.scoreDocs) {
          assertTrue(hit.doc <= maxAcceptedDoc);
          assertNotNull(vectors[hit.doc]);
          assertEquals(EUCLIDEAN.compare(queryVector, vectors[hit.doc]), hit.score, 1e-5f);
        }
      }
    }
  }

  /**
   * Asserts that a search with this shape reaches the branch that materializes the accepted
   * ordinals: the graph is searched rather than scanned, the searcher is the one optimized for
   * filtering, and materializing is cheaper than a lookup per test. A change to the numbers above
   * that breaks any of the three would otherwise leave that branch untested.
   */
  private static void assertGraphSearchMaterializesAcceptedOrds(
      int accepted, int numVectors, int k, int maxDoc) {
    int unfilteredVisit = HnswGraphSearcher.expectedVisitedNodes(k, numVectors);
    assertTrue(accepted + " <= " + unfilteredVisit, accepted > unfilteredVisit);
    assertTrue(KnnSearchStrategy.Hnsw.DEFAULT.useFilteredSearch((float) accepted / numVectors));
    assertTrue(
        Lucene99HnswVectorsReader.shouldMaterializeAcceptOrds(
            accepted, numVectors, unfilteredVisit, maxDoc));
  }

  /**
   * The decision to materialize the accepted ordinals weighs the words that a materialization
   * walks, which grow with the number of blocks the docs span and not with the number of accepted
   * docs, against the lookups that the filtered searcher is expected to make.
   */
  public void testShouldMaterializeAcceptOrds() {
    int k = 100;
    // A 200k doc segment where 10% of the docs have no vector: every filter that reaches the
    // filtered searcher materializes, whether it accepts 2% or 50% of the docs.
    int maxDoc = 200_000;
    int graphSize = 180_000;
    int unfilteredVisit = HnswGraphSearcher.expectedVisitedNodes(k, graphSize);
    for (int accepted : new int[] {4_000, 10_000, 40_000, 100_000, 118_000}) {
      assertTrue(
          "accepted=" + accepted,
          Lucene99HnswVectorsReader.shouldMaterializeAcceptOrds(
              accepted, graphSize, unfilteredVisit, maxDoc));
    }
    // A 10M doc segment: a filter that accepts half of the docs does not, since the searcher is
    // expected to make a few thousand lookups while a materialization walks a few hundred thousand
    // words, and a filter that accepts 1% of them does, since the lookups grow with the inverse of
    // the accepted fraction while the words do not grow at all.
    maxDoc = 10_000_000;
    graphSize = 9_000_000;
    unfilteredVisit = HnswGraphSearcher.expectedVisitedNodes(k, graphSize);
    assertFalse(
        Lucene99HnswVectorsReader.shouldMaterializeAcceptOrds(
            5_000_000, graphSize, unfilteredVisit, maxDoc));
    assertTrue(
        Lucene99HnswVectorsReader.shouldMaterializeAcceptOrds(
            100_000, graphSize, unfilteredVisit, maxDoc));
    // A filter that accepts a handful of docs spans at most that many blocks
    assertTrue(
        Lucene99HnswVectorsReader.shouldMaterializeAcceptOrds(
            unfilteredVisit + 1, graphSize, unfilteredVisit, maxDoc));
  }

  /**
   * A search whose filter accepts no more docs than an approximate search is expected to visit
   * scores the accepted vectors rather than searching the graph. It has to return the nearest of
   * them exactly, to score every accepted vector once, and to score nothing else.
   */
  public void testExactSearchOfAcceptedVectors() throws IOException {
    int numDocs = 6000;
    int dimension = 8;
    int k = 10;
    int maxAcceptedDoc = 79;
    float[][] vectors = new float[numDocs][];
    int[] ordOfDoc = new int[numDocs];
    int numVectors = 0;
    for (int i = 0; i < numDocs; i++) {
      // Every tenth doc has no vector, so that vector ordinals and doc ids differ
      ordOfDoc[i] = -1;
      if (i % 10 != 3) {
        vectors[i] = randomVector(dimension);
        ordOfDoc[i] = numVectors++;
      }
    }
    float[] queryVector = randomVector(dimension);

    try (Directory sparseDir = newDirectory();
        Directory denseDir = newDirectory()) {
      indexVectors(sparseDir, vectors, true);
      indexVectors(denseDir, vectors, false);

      try (DirectoryReader sparseReader = DirectoryReader.open(sparseDir);
          DirectoryReader denseReader = DirectoryReader.open(denseDir)) {
        LeafReader sparseLeaf = sparseReader.leaves().get(0).reader();
        LeafReader denseLeaf = denseReader.leaves().get(0).reader();

        // The docs below maxAcceptedDoc that have a vector, and the same vectors in the dense index
        FixedBitSet sparseAccepted = new FixedBitSet(numDocs);
        FixedBitSet denseAccepted = new FixedBitSet(numVectors);
        List<Integer> acceptedDocs = new ArrayList<>();
        for (int doc = 0; doc <= maxAcceptedDoc; doc++) {
          if (vectors[doc] != null) {
            sparseAccepted.set(doc);
            denseAccepted.set(ordOfDoc[doc]);
            acceptedDocs.add(doc);
          }
        }
        int accepted = acceptedDocs.size();
        assertScansAcceptedVectors(accepted, numVectors, k);
        // Brute force over the accepted vectors, nearest first, which is what a scan returns
        acceptedDocs.sort(
            Comparator.comparingDouble(doc -> -EUCLIDEAN.compare(queryVector, vectors[doc])));

        // The visited limit is unlimited, so nothing here stops the scan early
        TopDocs sparseDocs =
            sparseLeaf.searchNearestVectors(
                "field",
                queryVector,
                k,
                AcceptDocs.fromLiveDocs(sparseAccepted, numDocs),
                Integer.MAX_VALUE);
        TopDocs denseDocs =
            denseLeaf.searchNearestVectors(
                "field",
                queryVector,
                k,
                AcceptDocs.fromLiveDocs(denseAccepted, numVectors),
                Integer.MAX_VALUE);
        assertExactHits(sparseDocs, k, acceptedDocs, null, queryVector, vectors);
        assertExactHits(denseDocs, k, acceptedDocs, ordOfDoc, queryVector, vectors);

        // Accepting the docs that have no vector as well scores the same vectors and no more: the
        // scan enumerates the accepted ordinals, and those docs have none
        FixedBitSet acceptedWithHoles = new FixedBitSet(numDocs);
        acceptedWithHoles.set(0, maxAcceptedDoc + 1);
        assertScansAcceptedVectors(maxAcceptedDoc + 1, numVectors, k);
        TopDocs docsWithHoles =
            sparseLeaf.searchNearestVectors(
                "field",
                queryVector,
                k,
                AcceptDocs.fromLiveDocs(acceptedWithHoles, numDocs),
                Integer.MAX_VALUE);
        assertExactHits(docsWithHoles, k, acceptedDocs, null, queryVector, vectors);

        // A visit limit stops the scan, which then reports the vectors it did score as a lower
        // bound rather than as a count of the accepted ones
        assertTrue(Lucene99HnswVectorsReader.EXHAUSTIVE_BULK_SCORE_ORDS < accepted);
        TopDocs limitedDocs =
            sparseLeaf.searchNearestVectors(
                "field", queryVector, k, AcceptDocs.fromLiveDocs(sparseAccepted, numDocs), 1);
        assertEquals(TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO, limitedDocs.totalHits.relation());
        assertEquals(
            Lucene99HnswVectorsReader.EXHAUSTIVE_BULK_SCORE_ORDS, limitedDocs.totalHits.value());
        assertEquals(k, limitedDocs.scoreDocs.length);
      }
    }
  }

  /**
   * The same properties over accept sets that are drawn at random rather than chosen: every
   * accepted vector is scored once, nothing else is, and the hits are the nearest of them.
   */
  public void testExactSearchOfRandomAcceptSets() throws IOException {
    int numDocs = 2000;
    int dimension = 6;
    int k = 5;
    float[][] vectors = new float[numDocs][];
    int numVectors = 0;
    for (int i = 0; i < numDocs; i++) {
      if (i % 7 != 2) {
        vectors[i] = randomVector(dimension);
        numVectors++;
      }
    }
    int maxAccepted = HnswGraphSearcher.expectedVisitedNodes(k, numVectors);
    try (Directory dir = newDirectory()) {
      indexVectors(dir, vectors, true);
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        for (int iter = 0; iter < 10; iter++) {
          float[] queryVector = randomVector(dimension);
          FixedBitSet acceptDocs = new FixedBitSet(numDocs);
          List<Integer> acceptedDocs = new ArrayList<>();
          // Accept as many docs as a scan can be relied on to run for, with and without a vector
          int accepted = TestUtil.nextInt(random(), 1, maxAccepted);
          while (acceptDocs.cardinality() < accepted) {
            int doc = random().nextInt(numDocs);
            if (acceptDocs.getAndSet(doc) == false && vectors[doc] != null) {
              acceptedDocs.add(doc);
            }
          }
          assertScansAcceptedVectors(accepted, numVectors, k);
          acceptedDocs.sort(
              Comparator.comparingDouble(doc -> -EUCLIDEAN.compare(queryVector, vectors[doc])));
          TopDocs topDocs =
              leaf.searchNearestVectors(
                  "field",
                  queryVector,
                  k,
                  AcceptDocs.fromLiveDocs(acceptDocs, numDocs),
                  Integer.MAX_VALUE);
          assertExactHits(
              topDocs, Math.min(k, acceptedDocs.size()), acceptedDocs, null, queryVector, vectors);
        }
      }
    }
  }

  /** The scan of a field of byte vectors, and of a field of float16 vectors. */
  public void testExactSearchOfByteAndFloat16Vectors() throws IOException {
    int numDocs = 1000;
    int dimension = 4;
    int k = 5;
    int maxAcceptedDoc = 29;
    byte[][] byteVectors = new byte[numDocs][];
    short[][] float16Vectors = new short[numDocs][];
    int numVectors = 0;
    for (int i = 0; i < numDocs; i++) {
      // Every fifth doc has no vector in either field
      if (i % 5 != 0) {
        byteVectors[i] = randomVector8(dimension);
        float16Vectors[i] = randomVector16(dimension);
        numVectors++;
      }
    }
    byte[] byteQuery = randomVector8(dimension);
    short[] float16Query = randomVector16(dimension);

    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = new IndexWriterConfig().setCodec(TestUtil.getDefaultCodec());
      iwc.setMaxBufferedDocs(numDocs + 1);
      iwc.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int i = 0; i < numDocs; i++) {
          Document doc = new Document();
          if (byteVectors[i] != null) {
            doc.add(new KnnByteVectorField("bytes", byteVectors[i], EUCLIDEAN));
            doc.add(new KnnFloat16VectorField("float16", float16Vectors[i], EUCLIDEAN));
          }
          w.addDocument(doc);
        }
        w.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        FixedBitSet acceptDocs = new FixedBitSet(numDocs);
        acceptDocs.set(0, maxAcceptedDoc + 1);
        List<Integer> acceptedDocs = new ArrayList<>();
        for (int doc = 0; doc <= maxAcceptedDoc; doc++) {
          if (byteVectors[doc] != null) {
            acceptedDocs.add(doc);
          }
        }
        assertScansAcceptedVectors(maxAcceptedDoc + 1, numVectors, k);

        acceptedDocs.sort(
            Comparator.comparingDouble(doc -> -EUCLIDEAN.compare(byteQuery, byteVectors[doc])));
        TopDocs byteDocs =
            leaf.searchNearestVectors(
                "bytes",
                byteQuery,
                k,
                AcceptDocs.fromLiveDocs(acceptDocs, numDocs),
                Integer.MAX_VALUE);
        assertEquals(TotalHits.Relation.EQUAL_TO, byteDocs.totalHits.relation());
        assertEquals(acceptedDocs.size(), byteDocs.totalHits.value());
        assertEquals(k, byteDocs.scoreDocs.length);
        for (int i = 0; i < k; i++) {
          int doc = acceptedDocs.get(i);
          assertEquals(doc, byteDocs.scoreDocs[i].doc);
          assertEquals(
              EUCLIDEAN.compare(byteQuery, byteVectors[doc]), byteDocs.scoreDocs[i].score, 1e-5f);
        }

        acceptedDocs.sort(
            Comparator.comparingDouble(
                doc -> -EUCLIDEAN.compare(float16Query, float16Vectors[doc])));
        TopDocs float16Docs =
            leaf.searchNearestVectors(
                "float16",
                float16Query,
                k,
                AcceptDocs.fromLiveDocs(acceptDocs, numDocs),
                Integer.MAX_VALUE);
        assertEquals(TotalHits.Relation.EQUAL_TO, float16Docs.totalHits.relation());
        assertEquals(acceptedDocs.size(), float16Docs.totalHits.value());
        assertEquals(k, float16Docs.scoreDocs.length);
        for (int i = 0; i < k; i++) {
          int doc = acceptedDocs.get(i);
          assertEquals(doc, float16Docs.scoreDocs[i].doc);
          assertEquals(
              EUCLIDEAN.compare(float16Query, float16Vectors[doc]),
              float16Docs.scoreDocs[i].score,
              1e-5f);
        }
      }
    }
  }

  /**
   * A scan of a segment with deletions, whose deleted docs keep their vector ordinal until the
   * segment is merged away, must not score them.
   */
  public void testExactSearchOfSegmentWithDeletions() throws IOException {
    int numDocs = 1000;
    int dimension = 4;
    int k = 5;
    int maxAcceptedDoc = 44;
    float[][] vectors = new float[numDocs][];
    int numVectors = 0;
    for (int i = 0; i < numDocs; i++) {
      // Every fifth doc has no vector, and every third doc is deleted below
      if (i % 5 != 0) {
        vectors[i] = randomVector(dimension);
        numVectors++;
      }
    }
    float[] queryVector = randomVector(dimension);

    try (Directory dir = newDirectory()) {
      // One segment from a single flush, and no merge policy so that the deletions below stay
      // deletions rather than being merged away at commit
      IndexWriterConfig iwc =
          new IndexWriterConfig()
              .setCodec(TestUtil.getDefaultCodec())
              .setMergePolicy(NoMergePolicy.INSTANCE);
      iwc.setMaxBufferedDocs(numDocs + 1);
      iwc.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int i = 0; i < numDocs; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
          if (vectors[i] != null) {
            doc.add(new KnnFloatVectorField("field", vectors[i], EUCLIDEAN));
          }
          w.addDocument(doc);
        }
        w.flush();
        for (int i = 0; i < numDocs; i += 3) {
          w.deleteDocuments(new Term("id", Integer.toString(i)));
        }
        w.commit();
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
          assertEquals(1, reader.leaves().size());
          LeafReader leaf = reader.leaves().get(0).reader();
          Bits liveDocs = leaf.getLiveDocs();
          assertNotNull(liveDocs);
          // The deleted docs still have a vector ordinal, so only the accept set keeps them out
          assertEquals(numVectors, leaf.getFloatVectorValues("field").size());

          FixedBitSet acceptDocs = new FixedBitSet(numDocs);
          List<Integer> acceptedDocs = new ArrayList<>();
          for (int doc = 0; doc <= maxAcceptedDoc; doc++) {
            if (liveDocs.get(doc)) {
              acceptDocs.set(doc);
              if (vectors[doc] != null) {
                acceptedDocs.add(doc);
              }
            }
          }
          assertScansAcceptedVectors(acceptDocs.cardinality(), numVectors, k);
          acceptedDocs.sort(
              Comparator.comparingDouble(doc -> -EUCLIDEAN.compare(queryVector, vectors[doc])));
          TopDocs topDocs =
              leaf.searchNearestVectors(
                  "field",
                  queryVector,
                  k,
                  AcceptDocs.fromLiveDocs(acceptDocs, numDocs),
                  Integer.MAX_VALUE);
          assertExactHits(topDocs, k, acceptedDocs, null, queryVector, vectors);
          for (ScoreDoc hit : topDocs.scoreDocs) {
            assertTrue("deleted doc " + hit.doc, liveDocs.get(hit.doc));
          }
        }
      }
    }
  }

  /**
   * The ordinals that a field enumerates as accepted are exactly the ones it accepts one at a time,
   * in increasing order, and a field where every doc has a vector enumerates the accepted docs
   * themselves.
   */
  public void testAcceptedOrdsIterator() throws IOException {
    int numDocs = 1000;
    int dimension = 4;
    float[][] vectors = new float[numDocs][];
    for (int i = 0; i < numDocs; i++) {
      vectors[i] = i % 7 == 2 ? null : randomVector(dimension);
    }
    try (Directory sparseDir = newDirectory();
        Directory denseDir = newDirectory()) {
      indexVectors(sparseDir, vectors, true);
      indexVectors(denseDir, vectors, false);
      try (DirectoryReader sparseReader = DirectoryReader.open(sparseDir);
          DirectoryReader denseReader = DirectoryReader.open(denseDir)) {
        FloatVectorValues sparseValues =
            sparseReader.leaves().get(0).reader().getFloatVectorValues("field");
        for (int iter = 0; iter < 5; iter++) {
          FixedBitSet acceptDocs = new FixedBitSet(numDocs);
          for (int doc = 0; doc < numDocs; doc++) {
            // Accepts docs with and without a vector, and never accepts them all
            if (random().nextInt(4) > 0) {
              acceptDocs.set(doc);
            }
          }
          Bits lazy = sparseValues.getAcceptOrds(acceptDocs);
          DocIdSetIterator acceptedOrds =
              sparseValues.acceptedOrdsIterator(
                  acceptDocs, new BitSetIterator(acceptDocs, acceptDocs.cardinality()));
          assertNotNull(acceptedOrds);
          int enumerated = 0;
          int previousOrd = -1;
          for (int ord = acceptedOrds.nextDoc();
              ord != DocIdSetIterator.NO_MORE_DOCS;
              ord = acceptedOrds.nextDoc()) {
            assertTrue(ord + " <= " + previousOrd, ord > previousOrd);
            assertTrue("ord=" + ord, lazy.get(ord));
            previousOrd = ord;
            enumerated++;
          }
          int accepted = 0;
          for (int ord = 0; ord < sparseValues.size(); ord++) {
            if (lazy.get(ord)) {
              accepted++;
            }
          }
          assertEquals(accepted, enumerated);
        }
        assertNull(sparseValues.acceptedOrdsIterator(null, DocIdSetIterator.all(numDocs)));

        // A dense field has nothing to enumerate: its ordinals are its docs
        FloatVectorValues denseValues =
            denseReader.leaves().get(0).reader().getFloatVectorValues("field");
        FixedBitSet acceptDocs = new FixedBitSet(denseValues.size());
        acceptDocs.set(0, denseValues.size());
        BitSetIterator acceptDocsIterator =
            new BitSetIterator(acceptDocs, acceptDocs.cardinality());
        assertSame(
            acceptDocsIterator, denseValues.acceptedOrdsIterator(acceptDocs, acceptDocsIterator));
        assertNull(denseValues.acceptedOrdsIterator(null, acceptDocsIterator));
      }
    }
  }

  /**
   * Asserts that a search of this shape scores the accepted vectors rather than searching the
   * graph, which is what the reader does when the filter accepts no more docs than an unfiltered
   * search is expected to visit. A change that moves the boundary would otherwise leave the scan
   * untested here.
   */
  private static void assertScansAcceptedVectors(int acceptedDocCount, int numVectors, int k) {
    int unfilteredVisit = HnswGraphSearcher.expectedVisitedNodes(k, numVectors);
    assertTrue(acceptedDocCount + " > " + unfilteredVisit, unfilteredVisit >= acceptedDocCount);
  }

  /**
   * Asserts that these hits are the nearest accepted vectors with their scores, that every accepted
   * vector was scored, and that nothing else was.
   *
   * @param nearestFirst the accepted docs that have a vector, nearest first
   * @param docInIndex the doc that holds the vector of a doc of {@code nearestFirst}, or {@code
   *     null} when they are the same doc
   */
  private static void assertExactHits(
      TopDocs topDocs,
      int k,
      List<Integer> nearestFirst,
      int[] docInIndex,
      float[] queryVector,
      float[][] vectors) {
    // Every accepted vector was scored exactly once, and nothing else was
    assertEquals(TotalHits.Relation.EQUAL_TO, topDocs.totalHits.relation());
    assertEquals(nearestFirst.size(), topDocs.totalHits.value());
    assertEquals(k, topDocs.scoreDocs.length);
    for (int i = 0; i < k; i++) {
      int doc = nearestFirst.get(i);
      assertEquals(docInIndex == null ? doc : docInIndex[doc], topDocs.scoreDocs[i].doc);
      assertEquals(EUCLIDEAN.compare(queryVector, vectors[doc]), topDocs.scoreDocs[i].score, 1e-5f);
    }
  }

  /** A random float16 vector, as a doc that has one holds it. */
  private static short[] randomVector16(int dimension) {
    float[] vector = randomVector(dimension);
    short[] float16Vector = new short[dimension];
    for (int i = 0; i < dimension; i++) {
      float16Vector[i] = Float.floatToFloat16(vector[i]);
    }
    return float16Vector;
  }

  /**
   * The accepted ordinals that a sparse field materializes must be the same bits as the ones it
   * computes lazily, whether or not the accepted docs have a vector, and whether they are
   * materialized a word at a time from a {@link FixedBitSet} or a doc at a time from any other
   * accepted docs.
   */
  public void testMaterializedAcceptOrds() throws IOException {
    int numDocs = 1000;
    int dimension = 4;
    float[][] vectors = new float[numDocs][];
    for (int i = 0; i < numDocs; i++) {
      vectors[i] = i % 7 == 2 ? null : randomVector(dimension);
    }
    try (Directory sparseDir = newDirectory();
        Directory denseDir = newDirectory()) {
      indexVectors(sparseDir, vectors, true);
      indexVectors(denseDir, vectors, false);
      try (DirectoryReader sparseReader = DirectoryReader.open(sparseDir);
          DirectoryReader denseReader = DirectoryReader.open(denseDir)) {
        FloatVectorValues sparseValues =
            sparseReader.leaves().get(0).reader().getFloatVectorValues("field");
        for (int iter = 0; iter < 5; iter++) {
          FixedBitSet acceptDocs = new FixedBitSet(numDocs);
          SparseFixedBitSet sparseAcceptDocs = new SparseFixedBitSet(numDocs);
          for (int doc = 0; doc < numDocs; doc++) {
            // Accepts docs with and without a vector, and never accepts them all
            if (random().nextInt(4) > 0) {
              acceptDocs.set(doc);
              sparseAcceptDocs.set(doc);
            }
          }
          // Accept docs that are not a bit set are leap frogged a doc at a time
          Bits plainAcceptDocs =
              new Bits() {
                @Override
                public boolean get(int index) {
                  return acceptDocs.get(index);
                }

                @Override
                public int length() {
                  return acceptDocs.length();
                }
              };
          Bits lazy = sparseValues.getAcceptOrds(acceptDocs);
          Bits materialized =
              sparseValues.getAcceptOrds(
                  acceptDocs, new BitSetIterator(acceptDocs, acceptDocs.cardinality()));
          Bits sparseMaterialized =
              sparseValues.getAcceptOrds(
                  sparseAcceptDocs,
                  new BitSetIterator(sparseAcceptDocs, sparseAcceptDocs.cardinality()));
          Bits leapFrogged =
              sparseValues.getAcceptOrds(
                  plainAcceptDocs, new BitSetIterator(acceptDocs, acceptDocs.cardinality()));
          assertNotSame(lazy, materialized);
          assertNotSame(lazy, sparseMaterialized);
          assertNotSame(lazy, leapFrogged);
          assertEquals(sparseValues.size(), lazy.length());
          assertEquals(sparseValues.size(), materialized.length());
          assertEquals(sparseValues.size(), sparseMaterialized.length());
          assertEquals(sparseValues.size(), leapFrogged.length());
          for (int ord = 0; ord < sparseValues.size(); ord++) {
            assertEquals("ord=" + ord, lazy.get(ord), materialized.get(ord));
            assertEquals("ord=" + ord, lazy.get(ord), sparseMaterialized.get(ord));
            assertEquals("ord=" + ord, lazy.get(ord), leapFrogged.get(ord));
          }
          assertEquals(materialized, sparseMaterialized);
          assertEquals(materialized, leapFrogged);
        }
        assertNull(sparseValues.getAcceptOrds(null, DocIdSetIterator.all(numDocs)));

        // A dense field has nothing to materialize: its ordinals are its docs
        FloatVectorValues denseValues =
            denseReader.leaves().get(0).reader().getFloatVectorValues("field");
        FixedBitSet acceptDocs = new FixedBitSet(denseValues.size());
        acceptDocs.set(0, denseValues.size());
        assertSame(
            acceptDocs,
            denseValues.getAcceptOrds(
                acceptDocs, new BitSetIterator(acceptDocs, acceptDocs.cardinality())));
      }
    }
  }

  private void indexVectors(Directory dir, float[][] vectors, boolean indexDocsWithoutVectors)
      throws IOException {
    IndexWriterConfig iwc = new IndexWriterConfig().setCodec(TestUtil.getDefaultCodec());
    // Buffer every doc, so that both indexes are written as a single segment from the same
    // sequence of vectors and get the same graph
    iwc.setMaxBufferedDocs(vectors.length + 1);
    iwc.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    try (IndexWriter w = new IndexWriter(dir, iwc)) {
      for (float[] vector : vectors) {
        if (vector == null && indexDocsWithoutVectors == false) {
          continue;
        }
        Document doc = new Document();
        if (vector != null) {
          doc.add(new KnnFloatVectorField("field", vector, EUCLIDEAN));
        }
        w.addDocument(doc);
      }
      w.forceMerge(1);
    }
  }
}
