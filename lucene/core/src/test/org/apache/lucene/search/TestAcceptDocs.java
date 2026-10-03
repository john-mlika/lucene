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
package org.apache.lucene.search;

import com.carrotsearch.randomizedtesting.generators.RandomPicks;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOUtils;

public class TestAcceptDocs extends LuceneTestCase {

  public void testValidation() {
    // iterator supplier must be non-null
    expectThrows(NullPointerException.class, () -> AcceptDocs.fromIteratorSupplier(null, null, 1));

    // iterator supplier may not produce null iterators
    expectThrows(
        NullPointerException.class,
        () -> AcceptDocs.fromIteratorSupplier(() -> null, null, 1).iterator());

    // Bits length != maxDoc
    expectThrows(
        IllegalArgumentException.class, () -> AcceptDocs.fromLiveDocs(new Bits.MatchNoBits(3), 4));
  }

  public void testIteratorIgnoresDeletedDocs() throws IOException {
    int maxDoc = 5;
    int deletedDoc = 3;
    FixedBitSet liveDocs = new FixedBitSet(maxDoc);
    liveDocs.set(0, liveDocs.length());
    liveDocs.clear(deletedDoc);

    Bits liveDocsBits = liveDocs.asReadOnlyBits();

    AcceptDocs bitsAcceptDocs = AcceptDocs.fromLiveDocs(liveDocsBits, maxDoc);
    AcceptDocs iteratorAcceptDocs =
        AcceptDocs.fromIteratorSupplier(() -> DocIdSetIterator.all(maxDoc), liveDocsBits, maxDoc);

    for (AcceptDocs acceptDocs : Arrays.asList(bitsAcceptDocs, iteratorAcceptDocs)) {
      Bits acceptBits = acceptDocs.bits();
      assertEquals(maxDoc, acceptBits.length());
      for (int i = 0; i < maxDoc; ++i) {
        assertEquals(i != deletedDoc, acceptBits.get(i));
      }

      DocIdSetIterator iterator = acceptDocs.iterator();
      for (int i = 0; i < maxDoc; ++i) {
        if (i != deletedDoc) {
          assertEquals(i, iterator.nextDoc());
        }
      }
      assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }
  }

  public void testIteratorIsNew() throws IOException {
    int maxDoc = 5;
    AcceptDocs bitsAcceptDocs = AcceptDocs.fromLiveDocs(null, maxDoc);
    AcceptDocs iteratorAcceptDocs =
        AcceptDocs.fromIteratorSupplier(() -> DocIdSetIterator.all(maxDoc), null, maxDoc);

    for (AcceptDocs acceptDocs : Arrays.asList(bitsAcceptDocs, iteratorAcceptDocs)) {
      DocIdSetIterator iterator = acceptDocs.iterator();
      assertEquals(-1, iterator.docID());
      iterator.nextDoc();
      iterator = acceptDocs.iterator();
      assertEquals(-1, iterator.docID());

      // Triggers lazy loading of matches into a bit set when created from an iterator
      acceptDocs.bits();

      iterator = acceptDocs.iterator();
      assertEquals(-1, iterator.docID());
      iterator.nextDoc();
      iterator = acceptDocs.iterator();
      assertEquals(-1, iterator.docID());
    }
  }

  /**
   * A {@link BitSetIterator} that counts calls to {@link #nextDoc()}. It extends {@link
   * BitSetIterator} rather than wrapping one so that the bulk {@link BitSetIterator#intoBitSet}
   * implementation remains visible to callers.
   */
  private static class CountingBitSetIterator extends BitSetIterator {

    int nextDocCalls;

    CountingBitSetIterator(BitSet bits, long cost) {
      super(bits, cost);
    }

    @Override
    public int nextDoc() {
      nextDocCalls++;
      return super.nextDoc();
    }
  }

  /**
   * Dense accept docs must be loaded through {@link DocIdSetIterator#intoBitSet}, which {@link
   * BitSetIterator} implements by copying whole words, and only then be masked with live docs via
   * {@link Bits#applyMask}. Filtering the iterator upfront would fall back to one {@link
   * DocIdSetIterator#nextDoc()} call per matching doc. Both approaches return the same bits, so
   * this can only be observed by counting calls on the source iterator.
   */
  public void testDenseIteratorIsConsumedInBulkWhenSegmentHasDeletions() throws IOException {
    int iters = atLeast(20);
    for (int iter = 0; iter < iters; ++iter) {
      int maxDoc = TestUtil.nextInt(random(), 1024, 1 << 16);

      // Match every `step`th doc, so that the cost is always >= maxDoc >> 7 and AcceptDocs loads
      // matches into a FixedBitSet.
      FixedBitSet filter = new FixedBitSet(maxDoc);
      int step = TestUtil.nextInt(random(), 1, 16);
      for (int doc = random().nextInt(step); doc < maxDoc; doc += step) {
        filter.set(doc);
      }
      int filterCardinality = filter.cardinality();
      assertTrue(filterCardinality > (maxDoc >> 7));

      // A segment that has deletions, ie. a non-null liveDocs: none, one, or many deleted docs.
      FixedBitSet liveDocs = new FixedBitSet(maxDoc);
      liveDocs.set(0, maxDoc);
      int deleteCount =
          switch (random().nextInt(3)) {
            case 0 -> 0;
            case 1 -> 1;
            default -> TestUtil.nextInt(random(), 2, maxDoc);
          };
      for (int i = 0; i < deleteCount; ++i) {
        liveDocs.clear(random().nextInt(maxDoc));
      }

      CountingBitSetIterator iterator = new CountingBitSetIterator(filter, filterCardinality);
      int[] iteratorsPulled = new int[1];
      AcceptDocs acceptDocs =
          AcceptDocs.fromIteratorSupplier(
              () -> {
                iteratorsPulled[0]++;
                return iterator;
              },
              liveDocs.asReadOnlyBits(),
              maxDoc);

      Bits acceptBits = acceptDocs.bits();

      // The bit set must be built from a single bulk pass over the source iterator.
      assertEquals(1, iteratorsPulled[0]);
      assertTrue(
          "iterator was consumed one doc at a time: "
              + iterator.nextDocCalls
              + " nextDoc() calls for "
              + filterCardinality
              + " matching docs",
          iterator.nextDocCalls <= 2);

      // Loading in bulk must not change the result.
      FixedBitSet expected = filter.clone();
      expected.and(liveDocs);
      assertEquals(expected, acceptBits);
      assertEquals(expected.cardinality(), acceptDocs.cost());
    }
  }

  /**
   * The bits exposed by {@link AcceptDocs#bits()} are always the matches of the iterator
   * intersected with live docs, on both sides of the {@code maxDoc >> 7} dense/sparse boundary and
   * whatever the number of deleted docs.
   */
  public void testRandomBitsAreMatchesIntersectedWithLiveDocs() throws IOException {
    int iters = atLeast(100);
    for (int iter = 0; iter < iters; ++iter) {
      int maxDoc = TestUtil.nextInt(random(), 1, 5000);
      int threshold = maxDoc >> 7; // AcceptDocs' dense/sparse boundary

      int targetCardinality =
          switch (random().nextInt(6)) {
            case 0 -> 0;
            case 1 -> Math.max(0, threshold - 1);
            case 2 -> threshold;
            case 3 -> threshold + 1;
            case 4 -> maxDoc;
            default -> TestUtil.nextInt(random(), 0, maxDoc);
          };
      targetCardinality = Math.min(targetCardinality, maxDoc);

      FixedBitSet filter = new FixedBitSet(maxDoc);
      for (int cardinality = 0; cardinality < targetCardinality; ) {
        if (filter.getAndSet(random().nextInt(maxDoc)) == false) {
          cardinality++;
        }
      }
      long cost = filter.cardinality();

      // None, one or many deleted docs -- and sometimes a segment with no deletions at all.
      FixedBitSet liveDocs = new FixedBitSet(maxDoc);
      liveDocs.set(0, maxDoc);
      Bits liveDocsBits;
      if (random().nextInt(5) == 0) {
        liveDocsBits = null; // no deletions
      } else {
        int deleteCount =
            switch (random().nextInt(3)) {
              case 0 -> 0;
              case 1 -> 1;
              default -> TestUtil.nextInt(random(), 1, maxDoc);
            };
        for (int i = 0; i < deleteCount; ++i) {
          liveDocs.clear(random().nextInt(maxDoc));
        }
        liveDocsBits = liveDocs.asReadOnlyBits();
      }

      FixedBitSet expected = filter.clone();
      expected.and(liveDocs);

      // Half of the time, hide the fact that the source is a BitSetIterator behind a
      // FilterDocIdSetIterator, which uses the default impl of intoBitSet, so that the generic
      // path gets the same coverage as the optimized one.
      boolean opaque = random().nextBoolean();
      AcceptDocs acceptDocs =
          AcceptDocs.fromIteratorSupplier(
              () -> {
                DocIdSetIterator iterator = new BitSetIterator(filter, cost);
                return opaque ? new FilterDocIdSetIterator(iterator) : iterator;
              },
              liveDocsBits,
              maxDoc);

      Bits acceptBits = acceptDocs.bits();
      assertEquals(maxDoc, acceptBits.length());
      for (int doc = 0; doc < maxDoc; ++doc) {
        assertEquals(
            "doc=" + doc + " maxDoc=" + maxDoc + " cardinality=" + cost,
            expected.get(doc),
            acceptBits.get(doc));
      }
      assertEquals(expected.cardinality(), acceptDocs.cost());
      if (acceptBits instanceof FixedBitSet fixedBitSet) {
        assertEquals(expected, fixedBitSet);
      }

      // The iterator must agree with the bits.
      DocIdSetIterator iterator = acceptDocs.iterator();
      for (int doc = expected.nextSetBit(0);
          doc != DocIdSetIterator.NO_MORE_DOCS;
          doc = doc + 1 >= maxDoc ? DocIdSetIterator.NO_MORE_DOCS : expected.nextSetBit(doc + 1)) {
        assertEquals(doc, iterator.nextDoc());
      }
      assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }
  }

  /**
   * {@link AcceptDocs#fromScorerSupplier} accepts exactly the documents that {@link
   * AcceptDocs#fromIteratorSupplier} accepts over the same filter, whichever way it collects them:
   * random boolean filters over several segments with deletions, including nested disjunctions,
   * exclusions that may be two-phase and filters the query cache serves as bit sets.
   */
  public void testScorerSupplierAcceptsTheSameDocsAsTheIterator() throws IOException {
    try (Directory dir = newDirectory();
        RandomIndexWriter w = new RandomIndexWriter(random(), dir)) {
      int numDocs = atLeast(10_000);
      for (int i = 0; i < numDocs; ++i) {
        w.addDocument(filterDocument(i));
      }
      // At most two segments, so that one of them has at least one window of doc IDs.
      w.forceMerge(TestUtil.nextInt(random(), 1, 2));
      w.setDoRandomForceMerge(false);
      int numDeletes = TestUtil.nextInt(random(), 1, numDocs / 10);
      for (int i = 0; i < numDeletes; ++i) {
        w.deleteDocuments(new Term("id", Integer.toString(random().nextInt(numDocs))));
      }
      try (IndexReader reader = w.getReader()) {
        assertTrue(reader.hasDeletions());
        // No asserting wrappers: they wrap every scorer supplier, which would hide the boolean
        // filter's own supplier and therefore its bulk scorer.
        IndexSearcher searcher = newSearcher(reader, true, false);
        int iters = atLeast(30);
        for (int iter = 0; iter < iters; ++iter) {
          // The first filter is a dense conjunction with a two-phase exclusion, which is collected
          // with its bulk scorer; the others are random.
          Query filter =
              searcher.rewrite(
                  iter == 0
                      ? new BooleanQuery.Builder()
                          .add(new TermQuery(new Term("half", "0")), Occur.FILTER)
                          .add(new TermQuery(new Term("eighth", "2")), Occur.FILTER)
                          .add(
                              new IndexOrDocValuesQuery(
                                  IntPoint.newRangeQuery("num", 0, 99),
                                  NumericDocValuesField.newSlowRangeQuery("num", 0, 99)),
                              Occur.MUST_NOT)
                          .build()
                      : randomFilter(2));
          // No cache, a cache of every clause but the filter itself (a filter seen for the first
          // time), or a cache of everything (the filter is served as a bit set).
          int cacheMode = iter == 0 ? random().nextInt(2) : random().nextInt(3);
          LRUQueryCache queryCache =
              cacheMode == 0
                  ? null
                  : new LRUQueryCache(1000, 1 << 24, _ -> true, Float.POSITIVE_INFINITY);
          searcher.setQueryCache(queryCache);
          searcher.setQueryCachingPolicy(
              new QueryCachingPolicy() {
                @Override
                public void onUse(Query query) {}

                @Override
                public boolean shouldCache(Query query) {
                  return cacheMode == 2 || query.equals(filter) == false;
                }
              });
          Weight weight = searcher.createWeight(filter, ScoreMode.COMPLETE_NO_SCORES, 1f);
          for (LeafReaderContext ctx : searcher.getIndexReader().leaves()) {
            Bits liveDocs = ctx.reader().getLiveDocs();
            int maxDoc = ctx.reader().maxDoc();
            AcceptDocs expected =
                AcceptDocs.fromIteratorSupplier(() -> iterator(weight, ctx), liveDocs, maxDoc);
            AcceptDocs actual =
                AcceptDocs.fromScorerSupplier(() -> weight.scorerSupplier(ctx), liveDocs, maxDoc);
            String message = filter + " cacheMode=" + cacheMode + " in " + ctx;
            Bits expectedBits = expected.bits();
            Bits actualBits = actual.bits();
            assertEquals(message, maxDoc, actualBits.length());
            for (int doc = 0; doc < maxDoc; ++doc) {
              if (expectedBits.get(doc) != actualBits.get(doc)) {
                fail(message + " doc=" + doc + " expected=" + expectedBits.get(doc));
              }
            }
            assertEquals(message, expected.cost(), actual.cost());
            DocIdSetIterator expectedIterator = expected.iterator();
            DocIdSetIterator actualIterator = actual.iterator();
            for (int doc = expectedIterator.nextDoc();
                doc != DocIdSetIterator.NO_MORE_DOCS;
                doc = expectedIterator.nextDoc()) {
              assertEquals(message, doc, actualIterator.nextDoc());
            }
            assertEquals(message, DocIdSetIterator.NO_MORE_DOCS, actualIterator.nextDoc());
          }
          IOUtils.close(queryCache);
        }
      }
    }
  }

  /**
   * A dense conjunction is collected with the filter's bulk scorer. The clauses' iterators throw
   * from {@link DocIdSetIterator#nextDoc()}, which advancing the conjunction one document at a time
   * needs and the windowed bulk scorer does not, so collecting the same filter through its iterator
   * fails.
   */
  public void testDenseConjunctionIsCollectedWithItsBulkScorer() throws IOException {
    try (Directory dir = newDirectory()) {
      int numDocs = TestUtil.nextInt(random(), DenseConjunctionBulkScorer.WINDOW_SIZE, 20_000);
      indexSingleSegment(dir, numDocs);
      try (IndexReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        Query half = new TermQuery(new Term("half", "0"));
        Query eighth = new TermQuery(new Term("eighth", "2"));
        Query filter =
            new BooleanQuery.Builder()
                .add(new NoNextDocQuery(half), Occur.FILTER)
                .add(new NoNextDocQuery(eighth), Occur.FILTER)
                .build();
        Weight weight =
            searcher.createWeight(searcher.rewrite(filter), ScoreMode.COMPLETE_NO_SCORES, 1f);
        LeafReaderContext ctx = reader.leaves().get(0);
        int maxDoc = ctx.reader().maxDoc();

        AcceptDocs acceptDocs =
            AcceptDocs.fromScorerSupplier(() -> weight.scorerSupplier(ctx), null, maxDoc);
        Query plain =
            new BooleanQuery.Builder().add(half, Occur.FILTER).add(eighth, Occur.FILTER).build();
        Weight plainWeight =
            searcher.createWeight(searcher.rewrite(plain), ScoreMode.COMPLETE_NO_SCORES, 1f);
        AcceptDocs expected =
            AcceptDocs.fromIteratorSupplier(() -> iterator(plainWeight, ctx), null, maxDoc);
        assertEquals(expected.bits(), acceptDocs.bits());
        assertEquals(expected.cost(), acceptDocs.cost());
        assertTrue(acceptDocs.cost() > 0);

        AcceptDocs iteratorAcceptDocs =
            AcceptDocs.fromIteratorSupplier(() -> iterator(weight, ctx), null, maxDoc);
        expectThrows(UnsupportedOperationException.class, iteratorAcceptDocs::cost);
      }
    }
  }

  /**
   * A boolean filter that matches fewer than one document in {@link
   * DenseConjunctionBulkScorer#DENSITY_THRESHOLD_INVERSE} keeps its iterator, even though its
   * accept docs are dense enough to be stored in a {@link FixedBitSet}. The excluded clause's
   * iterator throws from {@link DocIdSetIterator#intoBitSet}, which excluding documents window by
   * window needs and the iterator does not. Dense filters with the same exclusion, with one or two
   * required clauses, show that the exclusion is loaded into a bit set when a filter is collected
   * with its bulk scorer.
   */
  public void testSparseFilterIsCollectedWithItsIterator() throws IOException {
    try (Directory dir = newDirectory()) {
      int numDocs = TestUtil.nextInt(random(), DenseConjunctionBulkScorer.WINDOW_SIZE, 20_000);
      indexSingleSegment(dir, numDocs);
      try (IndexReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        LeafReaderContext ctx = reader.leaves().get(0);
        int maxDoc = ctx.reader().maxDoc();
        Query exclusion = new NoIntoBitSetQuery(new TermQuery(new Term("eighth", "0")));

        // 1 doc in 64: dense enough for a FixedBitSet (maxDoc >> 7), not for dense windows
        // (maxDoc / DenseConjunctionBulkScorer.DENSITY_THRESHOLD_INVERSE).
        Query sparse =
            new BooleanQuery.Builder()
                .add(new TermQuery(new Term("sixtyfourth", "3")), Occur.FILTER)
                .add(new TermQuery(new Term("half", "1")), Occur.FILTER)
                .add(exclusion, Occur.MUST_NOT)
                .build();
        Weight sparseWeight =
            searcher.createWeight(searcher.rewrite(sparse), ScoreMode.COMPLETE_NO_SCORES, 1f);
        AcceptDocs acceptDocs =
            AcceptDocs.fromScorerSupplier(() -> sparseWeight.scorerSupplier(ctx), null, maxDoc);
        Query plainSparse =
            new BooleanQuery.Builder()
                .add(new TermQuery(new Term("sixtyfourth", "3")), Occur.FILTER)
                .add(new TermQuery(new Term("half", "1")), Occur.FILTER)
                .add(new TermQuery(new Term("eighth", "0")), Occur.MUST_NOT)
                .build();
        Weight plainWeight =
            searcher.createWeight(searcher.rewrite(plainSparse), ScoreMode.COMPLETE_NO_SCORES, 1f);
        AcceptDocs expected =
            AcceptDocs.fromIteratorSupplier(() -> iterator(plainWeight, ctx), null, maxDoc);
        assertEquals(expected.bits(), acceptDocs.bits());
        assertEquals(expected.cost(), acceptDocs.cost());
        assertTrue(acceptDocs.cost() > 0);

        Query dense =
            new BooleanQuery.Builder()
                .add(new TermQuery(new Term("half", "1")), Occur.FILTER)
                .add(new TermQuery(new Term("eighth", "3")), Occur.FILTER)
                .add(exclusion, Occur.MUST_NOT)
                .build();
        Query denseWithOneRequiredClause =
            new BooleanQuery.Builder()
                .add(new TermQuery(new Term("half", "1")), Occur.FILTER)
                .add(exclusion, Occur.MUST_NOT)
                .build();
        for (Query query : List.of(dense, denseWithOneRequiredClause)) {
          Weight denseWeight =
              searcher.createWeight(searcher.rewrite(query), ScoreMode.COMPLETE_NO_SCORES, 1f);
          AcceptDocs denseAcceptDocs =
              AcceptDocs.fromScorerSupplier(() -> denseWeight.scorerSupplier(ctx), null, maxDoc);
          expectThrows(UnsupportedOperationException.class, denseAcceptDocs::cost);
        }
      }
    }
  }

  /**
   * A filter that is not a boolean query, such as one the query cache serves as a bit set, keeps
   * its iterator, so that a bit set without deleted documents is used as is.
   */
  public void testBitSetFilterIsReused() throws IOException {
    int maxDoc = TestUtil.nextInt(random(), DenseConjunctionBulkScorer.WINDOW_SIZE, 1 << 16);
    FixedBitSet filter = new FixedBitSet(maxDoc);
    for (int doc = random().nextInt(2); doc < maxDoc; doc += 2) {
      filter.set(doc);
    }
    int cardinality = filter.cardinality();
    ScorerSupplier scorerSupplier =
        new ScorerSupplier() {
          @Override
          public Scorer get(long leadCost) {
            return new ConstantScoreScorer(
                0f, ScoreMode.COMPLETE_NO_SCORES, new BitSetIterator(filter, cardinality));
          }

          @Override
          public long cost() {
            return cardinality;
          }
        };
    AcceptDocs acceptDocs = AcceptDocs.fromScorerSupplier(() -> scorerSupplier, null, maxDoc);
    assertSame(filter, acceptDocs.bits());
    assertEquals(cardinality, acceptDocs.cost());
  }

  /**
   * {@link AcceptDocs#collectWithBulkScorer} takes a conjunctive boolean filter from the density at
   * which {@link BooleanScorerSupplier} evaluates conjunctions over windows of doc IDs, and nothing
   * else: not a filter one document sparser, not a pure disjunction, not a single query, not a
   * segment smaller than one window.
   */
  public void testCollectWithBulkScorer() throws IOException {
    int maxDoc = 2 * DenseConjunctionBulkScorer.WINDOW_SIZE;
    int dense = maxDoc / DenseConjunctionBulkScorer.DENSITY_THRESHOLD_INVERSE;
    try (Directory dir = newDirectory()) {
      try (IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
        for (int i = 0; i < maxDoc; ++i) {
          Document doc = new Document();
          doc.add(new StringField("all", "1", Store.NO));
          if (i < dense) {
            doc.add(new StringField("lead", "dense", Store.NO));
          }
          if (i < dense - 1) {
            doc.add(new StringField("lead", "sparser", Store.NO));
          }
          if (i % 8 == 0) {
            doc.add(new StringField("excluded", "1", Store.NO));
          }
          w.addDocument(doc);
        }
        w.forceMerge(1);
      }
      try (IndexReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        LeafReaderContext ctx = reader.leaves().get(0);
        assertEquals(maxDoc, ctx.reader().maxDoc());
        Query all = new TermQuery(new Term("all", "1"));
        Query excluded = new TermQuery(new Term("excluded", "1"));
        BiFunction<Query, Occur, Query> withAll =
            (lead, occur) -> new BooleanQuery.Builder().add(lead, occur).add(all, occur).build();
        Query denseLead = new TermQuery(new Term("lead", "dense"));
        Query sparserLead = new TermQuery(new Term("lead", "sparser"));
        assertTrue(collectWithBulkScorer(searcher, ctx, withAll.apply(denseLead, Occur.FILTER)));
        assertFalse(collectWithBulkScorer(searcher, ctx, withAll.apply(sparserLead, Occur.FILTER)));
        assertTrue(
            collectWithBulkScorer(
                searcher,
                ctx,
                new BooleanQuery.Builder()
                    .add(all, Occur.FILTER)
                    .add(excluded, Occur.MUST_NOT)
                    .build()));
        assertFalse(collectWithBulkScorer(searcher, ctx, withAll.apply(denseLead, Occur.SHOULD)));
        assertFalse(collectWithBulkScorer(searcher, ctx, all));
      }
    }
    try (Directory dir = newDirectory()) {
      indexSingleSegment(dir, DenseConjunctionBulkScorer.WINDOW_SIZE - 1);
      try (IndexReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        Query denseConjunction =
            new BooleanQuery.Builder()
                .add(new TermQuery(new Term("half", "0")), Occur.FILTER)
                .add(new TermQuery(new Term("eighth", "2")), Occur.FILTER)
                .build();
        assertFalse(collectWithBulkScorer(searcher, reader.leaves().get(0), denseConjunction));
      }
    }
  }

  private static boolean collectWithBulkScorer(
      IndexSearcher searcher, LeafReaderContext ctx, Query query) throws IOException {
    Weight weight =
        searcher.createWeight(searcher.rewrite(query), ScoreMode.COMPLETE_NO_SCORES, 1f);
    return AcceptDocs.collectWithBulkScorer(weight.scorerSupplier(ctx), ctx.reader().maxDoc());
  }

  private static Document filterDocument(int i) {
    Document doc = new Document();
    doc.add(new StringField("id", Integer.toString(i), Store.NO));
    doc.add(new StringField("half", Integer.toString(i % 2), Store.NO));
    doc.add(new StringField("eighth", Integer.toString(i % 8), Store.NO));
    doc.add(new StringField("sixtyfourth", Integer.toString(i % 64), Store.NO));
    int rare = random().nextInt(128);
    doc.add(new StringField("rare", Integer.toString(rare), Store.NO));
    int num = random().nextInt(1000);
    doc.add(new IntPoint("num", num));
    doc.add(new NumericDocValuesField("num", num));
    return doc;
  }

  private static void indexSingleSegment(Directory dir, int numDocs) throws IOException {
    try (IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
      for (int i = 0; i < numDocs; ++i) {
        w.addDocument(filterDocument(i));
      }
      w.forceMerge(1);
    }
  }

  private static Query randomFilter(int depth) {
    BooleanQuery.Builder builder = new BooleanQuery.Builder();
    int numRequired = TestUtil.nextInt(random(), 1, 4);
    for (int i = 0; i < numRequired; ++i) {
      builder.add(
          randomClause(depth), RandomPicks.randomFrom(random(), List.of(Occur.FILTER, Occur.MUST)));
    }
    if (depth > 0 && random().nextInt(4) == 0) {
      // optional clauses with a minimum number that should match
      int numShould = TestUtil.nextInt(random(), 2, 4);
      for (int i = 0; i < numShould; ++i) {
        builder.add(randomClause(depth - 1), Occur.SHOULD);
      }
      builder.setMinimumNumberShouldMatch(TestUtil.nextInt(random(), 1, numShould - 1));
    }
    int numProhibited = random().nextInt(3);
    for (int i = 0; i < numProhibited; ++i) {
      builder.add(randomClause(0), Occur.MUST_NOT);
    }
    return builder.build();
  }

  private static Query randomClause(int depth) {
    int lower = random().nextInt(1000);
    int upper = lower + random().nextInt(1000);
    return switch (random().nextInt(depth > 0 ? 7 : 6)) {
      case 0 -> new TermQuery(new Term("half", Integer.toString(random().nextInt(2))));
      case 1 -> new TermQuery(new Term("eighth", Integer.toString(random().nextInt(8))));
      case 2 -> new TermQuery(new Term("sixtyfourth", Integer.toString(random().nextInt(64))));
      case 3 -> new TermQuery(new Term("rare", Integer.toString(random().nextInt(128))));
      case 4 -> IntPoint.newRangeQuery("num", lower, upper);
      // may be evaluated as a two-phase iterator over doc values
      case 5 ->
          new IndexOrDocValuesQuery(
              IntPoint.newRangeQuery("num", lower, upper),
              NumericDocValuesField.newSlowRangeQuery("num", lower, upper));
      default -> randomFilter(depth - 1);
    };
  }

  private static DocIdSetIterator iterator(Weight weight, LeafReaderContext ctx)
      throws IOException {
    Scorer scorer = weight.scorer(ctx);
    return scorer == null ? DocIdSetIterator.empty() : scorer.iterator();
  }

  /** A query whose iterators are rewrapped by {@link #wrap}. */
  private abstract static class IteratorWrappingQuery extends Query {
    final Query in;

    IteratorWrappingQuery(Query in) {
      this.in = in;
    }

    abstract DocIdSetIterator wrap(DocIdSetIterator iterator);

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost)
        throws IOException {
      Weight inner = in.createWeight(searcher, scoreMode, boost);
      return new FilterWeight(this, inner) {
        @Override
        public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
          ScorerSupplier supplier = inner.scorerSupplier(context);
          if (supplier == null) {
            return null;
          }
          return new ScorerSupplier() {
            @Override
            public Scorer get(long leadCost) throws IOException {
              return new ConstantScoreScorer(
                  0f, ScoreMode.COMPLETE_NO_SCORES, wrap(supplier.get(leadCost).iterator()));
            }

            @Override
            public long cost() throws IOException {
              return supplier.cost();
            }
          };
        }
      };
    }

    @Override
    public String toString(String field) {
      return getClass().getSimpleName() + "(" + in.toString(field) + ")";
    }

    @Override
    public void visit(QueryVisitor visitor) {
      in.visit(visitor.getSubVisitor(Occur.FILTER, this));
    }

    @Override
    public boolean equals(Object other) {
      return sameClassAs(other) && in.equals(((IteratorWrappingQuery) other).in);
    }

    @Override
    public int hashCode() {
      return 31 * classHash() + in.hashCode();
    }
  }

  /** Iterators that throw from nextDoc() and forward the bulk methods. */
  private static class NoNextDocQuery extends IteratorWrappingQuery {
    NoNextDocQuery(Query in) {
      super(in);
    }

    @Override
    DocIdSetIterator wrap(DocIdSetIterator iterator) {
      return new FilterDocIdSetIterator(iterator) {
        @Override
        public int nextDoc() {
          throw new UnsupportedOperationException();
        }

        @Override
        public void intoBitSet(int upTo, FixedBitSet bitSet, int offset) throws IOException {
          in.intoBitSet(upTo, bitSet, offset);
        }

        @Override
        public int docIDRunEnd() throws IOException {
          return in.docIDRunEnd();
        }
      };
    }
  }

  /** Iterators that throw from intoBitSet(). */
  private static class NoIntoBitSetQuery extends IteratorWrappingQuery {
    NoIntoBitSetQuery(Query in) {
      super(in);
    }

    @Override
    DocIdSetIterator wrap(DocIdSetIterator iterator) {
      return new FilterDocIdSetIterator(iterator) {
        @Override
        public void intoBitSet(int upTo, FixedBitSet bitSet, int offset) {
          throw new UnsupportedOperationException();
        }
      };
    }
  }
}
