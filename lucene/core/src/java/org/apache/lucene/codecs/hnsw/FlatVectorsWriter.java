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

import java.io.Closeable;
import java.io.IOException;
import java.util.function.IntPredicate;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.util.IORunnable;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;

/**
 * Vectors' writer for a field that allows additional indexing logic to be implemented by the caller
 *
 * @lucene.experimental
 */
public abstract class FlatVectorsWriter extends KnnVectorsWriter {
  /** Scorer for flat vectors */
  protected final FlatVectorsScorer vectorsScorer;

  /** Sole constructor */
  protected FlatVectorsWriter(FlatVectorsScorer vectorsScorer) {
    this.vectorsScorer = vectorsScorer;
  }

  /**
   * @return the {@link FlatVectorsScorer} for this reader.
   */
  public FlatVectorsScorer getFlatVectorScorer() {
    return vectorsScorer;
  }

  /**
   * Add a new field for indexing
   *
   * @param fieldInfo fieldInfo of the field to add
   * @return a writer for the field
   * @throws IOException if an I/O error occurs when adding the field
   */
  @Override
  public abstract FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException;

  @Override
  public final IORunnable mergeOneField(FieldInfo fieldInfo, MergeState mergeState)
      throws IOException {
    mergeOneFlatVectorField(fieldInfo, mergeState);
    return null;
  }

  public abstract void mergeOneFlatVectorField(FieldInfo fieldInfo, MergeState mergeState)
      throws IOException;

  /**
   * Merges one field as {@link #mergeOneFlatVectorField} does, and prepares what a scorer over the
   * merged vectors needs, so that the caller does not have to read them back to build one.
   *
   * <p>A writer that wraps another one should forward this method as well as {@link
   * #mergeOneFlatVectorField}: forwarding only the latter is correct, but the wrapped writer never
   * prepares anything. Forwarding pays off only where the wrapping format opens the wrapped
   * writer's own reader over the merged segment; otherwise {@link MergeScorerData#scorerSupplier}
   * refuses what was prepared and the merge falls back anyway.
   *
   * @param fieldInfo fieldInfo of the field to merge
   * @param mergeState the state of the merge
   * @param needsMergeScorer whether a merge scorer will be built over a merge of that many vectors;
   *     a writer tests it on the count it has before writing, which does not account for deletions
   *     and so can exceed what it merges, and prepares only if it returns {@code true}
   * @return what this writer prepared, for the caller to close, or {@code null} if it prepared
   *     nothing, which is what the default implementation returns and what any writer returns for a
   *     field it cannot prepare for; the caller then builds the scorer itself
   * @throws IOException if an I/O error occurs
   */
  public MergeScorerData mergeOneFlatVectorFieldForMergeScorer(
      FieldInfo fieldInfo, MergeState mergeState, IntPredicate needsMergeScorer)
      throws IOException {
    mergeOneFlatVectorField(fieldInfo, mergeState);
    return null;
  }

  /**
   * What a {@link FlatVectorsWriter} prepared, while it merged a field, for a scorer over the
   * vectors it merged. The caller closes it whether or not it takes a supplier from it.
   * Implementations need not be thread-safe: one merge thread uses them.
   */
  public interface MergeScorerData extends Closeable {

    /**
     * Returns a scorer supplier over the merged vectors, which the caller closes to release what
     * this data prepared, or {@code null} if this data cannot serve {@code mergedReader}, in which
     * case the caller builds a scorer itself. This data is matched to its writer's own reader by
     * type, so a reader that wraps that reader, even over the right segment, is one it cannot
     * serve.
     *
     * <p>Called at most once, and only once the writer that prepared this has been finished and
     * closed and {@code mergedReader} has been opened over the segment it wrote. A second call, or
     * a call after {@link #close}, throws {@link IllegalStateException}.
     *
     * @param mergedReader a reader over the segment the writer wrote
     * @return a scorer supplier over the merged vectors, or {@code null} if this data cannot serve
     *     that reader
     * @throws IOException if an I/O error occurs
     */
    CloseableRandomVectorScorerSupplier scorerSupplier(FlatVectorsReader mergedReader)
        throws IOException;

    /**
     * Releases what {@link #scorerSupplier} did not take over. Idempotent, and must not throw, so
     * that a caller releasing on the way out of a failed merge does not replace its exception.
     */
    @Override
    void close();
  }
}
