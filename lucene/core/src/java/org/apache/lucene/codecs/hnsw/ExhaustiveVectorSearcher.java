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
package org.apache.lucene.codecs.hnsw;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/**
 * Scores every accepted vector rather than searching a graph, which is exact rather than
 * approximate. This is what a format without a graph does, and what a format with one does when the
 * filter accepts few enough vectors that scoring all of them is the cheaper of the two.
 *
 * <p>Only the accepted ordinals are scored when {@link KnnVectorValues#materializeAcceptOrds(Bits,
 * long)} materializes them, which every format that ships with Lucene does for the bit sets that a
 * filtered {@link AcceptDocs} builds, as long as walking them costs less than testing every
 * ordinal. Otherwise every ordinal is tested against the accepted ordinals, which is also what a
 * search that accepts every doc, or only the live ones, does.
 *
 * @lucene.experimental
 */
public final class ExhaustiveVectorSearcher {

  /**
   * How many ordinals are scored in one call to {@link RandomVectorScorer#bulkScore(int[], float[],
   * int)}.
   */
  public static final int BULK_SCORE_ORDS = 64;

  private ExhaustiveVectorSearcher() {}

  /**
   * Scores the vectors that {@code acceptDocs} accepts, collecting into {@code knnCollector}.
   *
   * @param scorer the scorer to compare the query with the vectors
   * @param knnCollector the collector of the top hits, which also carries the visit limit
   * @param acceptDocs the accepted docs
   */
  public static void search(
      RandomVectorScorer scorer, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    Bits accepted = acceptDocs.bits();
    // Without a materialization the scan tests every ordinal
    BitSet materialized =
        accepted == null ? null : scorer.materializeAcceptOrds(accepted, scorer.maxOrd());
    search(
        scorer,
        knnCollector,
        materialized != null ? materialized : scorer.getAcceptOrds(accepted),
        materialized);
  }

  /**
   * Scores the accepted vectors, for a caller that has already asked the scorer for the accepted
   * ordinals: exactly the ordinals of {@code materializedAcceptOrds} when there is one, every
   * ordinal tested against {@code acceptedOrds} otherwise.
   *
   * @param scorer the scorer to compare the query with the vectors
   * @param knnCollector the collector of the top hits, which also carries the visit limit
   * @param acceptedOrds the accepted ordinals, or {@code null} if all vectors are accepted
   * @param materializedAcceptOrds the accepted ordinals as a bit set to enumerate, or {@code null}
   *     if they were not materialized
   */
  public static void search(
      RandomVectorScorer scorer,
      KnnCollector knnCollector,
      Bits acceptedOrds,
      BitSet materializedAcceptOrds)
      throws IOException {
    int[] ords = new int[BULK_SCORE_ORDS];
    float[] scores = new float[BULK_SCORE_ORDS];
    int numOrds = 0;
    if (materializedAcceptOrds != null) {
      // Dense values hand back the accepted docs themselves, which may be a shorter bit set than
      // the ordinals; never a longer one, that would feed ordinals past the last vector
      assert materializedAcceptOrds.length() <= scorer.maxOrd()
          : "accepted ordinals of "
              + materializedAcceptOrds.length()
              + " bits for "
              + scorer.maxOrd()
              + " vectors";
      DocIdSetIterator it =
          new BitSetIterator(
              materializedAcceptOrds, materializedAcceptOrds.approximateCardinality());
      for (int ord = it.nextDoc(); ord != NO_MORE_DOCS; ord = it.nextDoc()) {
        if (knnCollector.earlyTerminated()) {
          break;
        }
        ords[numOrds++] = ord;
        if (numOrds == ords.length) {
          bulkScoreAndCollect(scorer, knnCollector, ords, scores, numOrds);
          numOrds = 0;
        }
      }
    } else {
      int numVectors = scorer.maxOrd();
      for (int ord = 0; ord < numVectors; ord++) {
        if (acceptedOrds == null || acceptedOrds.get(ord)) {
          if (knnCollector.earlyTerminated()) {
            break;
          }
          ords[numOrds++] = ord;
          if (numOrds == ords.length) {
            bulkScoreAndCollect(scorer, knnCollector, ords, scores, numOrds);
            numOrds = 0;
          }
        }
      }
    }
    if (numOrds > 0) {
      bulkScoreAndCollect(scorer, knnCollector, ords, scores, numOrds);
    }
  }

  private static void bulkScoreAndCollect(
      RandomVectorScorer scorer, KnnCollector knnCollector, int[] ords, float[] scores, int numOrds)
      throws IOException {
    knnCollector.incVisitedCount(numOrds);
    if (scorer.bulkScore(ords, scores, numOrds) > knnCollector.minCompetitiveSimilarity()) {
      for (int i = 0; i < numOrds; i++) {
        knnCollector.collect(scorer.ordToDoc(ords[i]), scores[i]);
      }
    }
  }
}
