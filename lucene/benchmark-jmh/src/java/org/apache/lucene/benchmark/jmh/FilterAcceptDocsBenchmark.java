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
package org.apache.lucene.benchmark.jmh;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexOrDocValuesQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LRUQueryCache;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryCachingPolicy;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.Bits;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Benchmarks building the accept docs of a filtered vector query, which collects every live
 * document that matches the filter into a bit set, over one segment of 1M documents with 5% of them
 * deleted.
 *
 * <p>The filter is a conjunction with exclusions, see {@link Shape}. Numeric clauses are {@link
 * IndexOrDocValuesQuery}s, so they may run as two-phase iterators over doc values.
 *
 * <p>Parameters explore:
 *
 * <ul>
 *   <li>{@code step} — the leading clause matches about one document in {@code step}: 256 is below
 *       the density at which accept docs are stored in a {@code FixedBitSet} (1/128), 100, 64 and
 *       40 are below the density at which a conjunction is evaluated over windows of doc IDs
 *       (1/32), 24, 8 and 2 are above it.
 *   <li>{@code shape} — the filter, see {@link Shape}.
 *   <li>{@code cache} — which clauses the query cache serves, see {@link Cache}.
 *   <li>{@code pollute} — whether other filters are built and counted before measuring, so that the
 *       iterator call sites see several implementations, as in a process that runs many kinds of
 *       queries.
 *   <li>{@code variant} — how the accept docs are built, see {@link Variant}.
 * </ul>
 *
 * <p>The index does not depend on the parameters; it is written once under {@code java.io.tmpdir}
 * and reused by later trials and forks.
 *
 * <p>Run with:
 *
 * <pre>
 *   ./gradlew -p lucene/benchmark-jmh assemble
 *   java -jar lucene/benchmark-jmh/build/benchmarks/lucene-benchmark-jmh-*.jar FilterAcceptDocsBenchmark
 * </pre>
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 2,
    jvmArgsAppend = {"-Xmx2g", "-Xms2g", "-XX:+AlwaysPreTouch"})
public class FilterAcceptDocsBenchmark {

  private static final int NUM_DOCS = 1_000_000;
  private static final int[] STEPS = {256, 100, 64, 40, 24, 8, 2};

  @Param({"256", "100", "64", "40", "24", "8", "2"})
  public int step;

  /** The filter. */
  public enum Shape {
    /**
     * FILTER(a term that matches one document in {@code step}, a term that matches half of the
     * documents, a numeric range that matches 80% of them) MUST_NOT(two numeric terms that match 2%
     * of the documents each).
     */
    TERMS,
    /**
     * FILTER(a set of numeric values that matches one document in {@code step}, a numeric term that
     * matches 70% of the documents, a disjunction of a numeric term that matches half of the
     * documents and a keyword term that matches 5% of them) MUST_NOT(two numeric terms that match
     * 2% of the documents each), the shape of a marketplace product filter.
     */
    NUMERIC_SET,
    /**
     * FILTER(a disjunction of a term that matches 1/8 of the documents and a term that matches
     * 1/64, a term that matches half of the documents, a numeric range that matches one document in
     * {@code step}) MUST_NOT(two terms that match 1/64 of the documents each).
     */
    RANGE_LED,
    /**
     * FILTER(a term that matches one document in {@code step}) MUST_NOT(two numeric terms that
     * match 2% of the documents each): one required clause.
     */
    ONE_REQUIRED,
    /**
     * SHOULD(a term that matches one document in {@code step}, a term that matches 1/64 of the
     * documents): a pure disjunction.
     */
    DISJUNCTION
  }

  @Param({"TERMS", "NUMERIC_SET", "RANGE_LED", "ONE_REQUIRED", "DISJUNCTION"})
  public Shape shape;

  @Param({"false", "true"})
  public boolean pollute;

  /** Which clauses the query cache serves. */
  public enum Cache {
    /** None, as for a filter whose clauses have not been seen yet. */
    NONE,
    /** Every clause but the filter itself, as for a new combination of common clauses. */
    CLAUSES
  }

  /** How the accept docs are built. */
  public enum Variant {
    /** {@link AcceptDocs#fromIteratorSupplier} over the filter's iterator. */
    ITERATOR,
    /** {@link AcceptDocs#fromScorerSupplier} over the filter's scorer supplier. */
    SCORER_SUPPLIER
  }

  @Param({"NONE", "CLAUSES"})
  public Cache cache;

  @Param({"ITERATOR", "SCORER_SUPPLIER"})
  public Variant variant;

  private Directory dir;
  private IndexReader reader;
  private LeafReaderContext context;
  private Bits liveDocs;
  private int maxDoc;
  private Weight filterWeight;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    Path path = Path.of(System.getProperty("java.io.tmpdir"), "lucene-filter-accept-docs-v3");
    Path done = path.resolve("done");
    if (Files.exists(done) == false) {
      Files.createDirectories(path);
      try (Directory d = MMapDirectory.open(path)) {
        writeIndex(d);
      }
      Files.createFile(done);
    }
    dir = MMapDirectory.open(path);
    reader = DirectoryReader.open(dir);
    context = reader.leaves().get(0);
    liveDocs = context.reader().getLiveDocs();
    maxDoc = context.reader().maxDoc();

    if (pollute) {
      pollute();
    }
    Query filter = filter(shape, step);
    IndexSearcher searcher = new IndexSearcher(reader);
    Query rewritten = searcher.rewrite(filter);
    switch (cache) {
      case NONE -> searcher.setQueryCache(null);
      case CLAUSES -> {
        searcher.setQueryCache(
            new LRUQueryCache(100, 64L << 20, _ -> true, Float.POSITIVE_INFINITY));
        searcher.setQueryCachingPolicy(
            new QueryCachingPolicy() {
              @Override
              public void onUse(Query query) {}

              @Override
              public boolean shouldCache(Query query) {
                return query.equals(rewritten) == false;
              }
            });
      }
    }
    filterWeight = searcher.createWeight(rewritten, ScoreMode.COMPLETE_NO_SCORES, 1f);
    // Warm the clause cache, and check that both variants accept the same number of documents.
    int count = build(Variant.ITERATOR);
    if (count == 0 || build(Variant.SCORER_SUPPLIER) != count || build(variant) != count) {
      throw new IllegalStateException("variants disagree on " + count + " documents");
    }
  }

  private static Query filter(Shape shape, int step) {
    return switch (shape) {
      case TERMS ->
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("step" + step, "1")), Occur.FILTER)
              .add(new TermQuery(new Term("half", "1")), Occur.FILTER)
              .add(numericRange("num", 0, 799), Occur.FILTER)
              .add(numericRange("flag", 21, 21), Occur.MUST_NOT)
              .add(numericRange("flag", 22, 22), Occur.MUST_NOT)
              .build();
      case ONE_REQUIRED ->
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("step" + step, "1")), Occur.FILTER)
              .add(numericRange("flag", 21, 21), Occur.MUST_NOT)
              .add(numericRange("flag", 22, 22), Occur.MUST_NOT)
              .build();
      case DISJUNCTION ->
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("step" + step, "1")), Occur.SHOULD)
              .add(new TermQuery(new Term("rare", "3")), Occur.SHOULD)
              .build();
      case RANGE_LED -> {
        Query disjunction =
            new BooleanQuery.Builder()
                .add(new TermQuery(new Term("mid", "1")), Occur.SHOULD)
                .add(new TermQuery(new Term("rare", "3")), Occur.SHOULD)
                .build();
        yield new BooleanQuery.Builder()
            .add(disjunction, Occur.FILTER)
            .add(new TermQuery(new Term("half", "1")), Occur.FILTER)
            .add(IntPoint.newRangeQuery("num", 0, 1000 / step - 1), Occur.FILTER)
            .add(new TermQuery(new Term("rare", "5")), Occur.MUST_NOT)
            .add(new TermQuery(new Term("rare", "7")), Occur.MUST_NOT)
            .build();
      }
      case NUMERIC_SET -> {
        int[] types = new int[Math.round(1000f / step)];
        for (int i = 0; i < types.length; i++) {
          types[i] = i;
        }
        Query typeSet =
            new IndexOrDocValuesQuery(
                IntPoint.newSetQuery("type", types),
                SortedNumericDocValuesField.newSlowSetQuery("type", toLongs(types)));
        Query disjunction =
            new BooleanQuery.Builder()
                .add(numericRange("tagged", 0, 0), Occur.SHOULD)
                .add(new TermQuery(new Term("store", "7")), Occur.SHOULD)
                .build();
        yield new BooleanQuery.Builder()
            .add(typeSet, Occur.FILTER)
            .add(numericRange("visible", 0, 0), Occur.FILTER)
            .add(disjunction, Occur.FILTER)
            .add(numericRange("flag", 21, 21), Occur.MUST_NOT)
            .add(numericRange("flag", 22, 22), Occur.MUST_NOT)
            .build();
      }
    };
  }

  private static long[] toLongs(int[] values) {
    long[] longs = new long[values.length];
    for (int i = 0; i < values.length; i++) {
      longs[i] = values[i];
    }
    return longs;
  }

  /** Builds and counts filters of other shapes with no query cache. */
  private void pollute() throws IOException {
    IndexSearcher searcher = new IndexSearcher(reader);
    searcher.setQueryCache(null);
    List<Query> filters =
        List.of(
            filter(shape == Shape.TERMS ? Shape.NUMERIC_SET : Shape.TERMS, 8),
            filter(shape, 100),
            new TermQuery(new Term("half", "0")),
            numericRange("num", 100, 899),
            new BooleanQuery.Builder()
                .add(new TermQuery(new Term("store", "3")), Occur.SHOULD)
                .add(new TermQuery(new Term("step8", "1")), Occur.SHOULD)
                .build());
    for (int i = 0; i < 20; i++) {
      for (Query filter : filters) {
        Weight weight =
            searcher.createWeight(searcher.rewrite(filter), ScoreMode.COMPLETE_NO_SCORES, 1f);
        AcceptDocs.fromIteratorSupplier(() -> iterator(weight), liveDocs, maxDoc).cost();
        searcher.count(filter);
      }
    }
  }

  private DocIdSetIterator iterator(Weight weight) throws IOException {
    Scorer scorer = weight.scorer(context);
    return scorer == null ? DocIdSetIterator.empty() : scorer.iterator();
  }

  private static Query numericRange(String field, int lower, int upper) {
    return new IndexOrDocValuesQuery(
        IntPoint.newRangeQuery(field, lower, upper),
        NumericDocValuesField.newSlowRangeQuery(field, lower, upper));
  }

  private static void writeIndex(Directory d) throws IOException {
    Random random = new Random(42);
    try (IndexWriter w = new IndexWriter(d, new IndexWriterConfig().setRAMBufferSizeMB(256))) {
      for (int i = 0; i < NUM_DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("id", Integer.toString(i), Store.NO));
        for (int step : STEPS) {
          if (random.nextInt(step) == 0) {
            doc.add(new StringField("step" + step, "1", Store.NO));
          }
        }
        doc.add(new StringField("half", Integer.toString(random.nextInt(2)), Store.NO));
        int num = random.nextInt(1000);
        doc.add(new IntPoint("num", num));
        doc.add(new NumericDocValuesField("num", num));
        int flag = random.nextInt(50);
        doc.add(new IntPoint("flag", flag));
        doc.add(new NumericDocValuesField("flag", flag));
        int type = random.nextInt(1000);
        doc.add(new IntPoint("type", type));
        doc.add(new SortedNumericDocValuesField("type", type));
        int visible = random.nextInt(10) < 7 ? 0 : 1;
        doc.add(new IntPoint("visible", visible));
        doc.add(new NumericDocValuesField("visible", visible));
        int tagged = random.nextInt(2);
        doc.add(new IntPoint("tagged", tagged));
        doc.add(new NumericDocValuesField("tagged", tagged));
        doc.add(new StringField("store", Integer.toString(random.nextInt(20)), Store.NO));
        doc.add(new StringField("mid", Integer.toString(random.nextInt(8)), Store.NO));
        doc.add(new StringField("rare", Integer.toString(random.nextInt(64)), Store.NO));
        w.addDocument(doc);
      }
      w.forceMerge(1);
      for (int i = 0; i < NUM_DOCS / 20; i++) {
        w.deleteDocuments(new Term("id", Integer.toString(random.nextInt(NUM_DOCS))));
      }
      w.commit();
    }
  }

  @TearDown(Level.Trial)
  public void tearDown() throws IOException {
    reader.close();
    dir.close();
  }

  @Benchmark
  public int buildAcceptDocs() throws IOException {
    return build(variant);
  }

  private int build(Variant how) throws IOException {
    AcceptDocs acceptDocs =
        switch (how) {
          case ITERATOR ->
              AcceptDocs.fromIteratorSupplier(() -> iterator(filterWeight), liveDocs, maxDoc);
          case SCORER_SUPPLIER ->
              AcceptDocs.fromScorerSupplier(
                  () -> filterWeight.scorerSupplier(context), liveDocs, maxDoc);
        };
    return acceptDocs.cost();
  }
}
