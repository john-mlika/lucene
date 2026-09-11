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
   * merged vectors needs, so that the caller does not have to build one from those vectors once
   * they are written.
   *
   * <p>A writer that can prepare it tests {@code needsMergeScorer} on the number of vectors the
   * merge writes, prepares only if it returns {@code true}, and returns a handle to what it
   * prepared. Returning {@code null} means nothing was prepared, which is the default and is also
   * what a writer with nothing to prepare for this field returns; the caller then builds the scorer
   * itself. A writer that wraps another one should forward this method as well as {@link
   * #mergeOneFlatVectorField}: forwarding only the latter is correct, but the wrapped writer never
   * prepares anything.
   *
   * @param fieldInfo fieldInfo of the field to merge
   * @param mergeState the state of the merge
   * @param needsMergeScorer whether a merge scorer will be built over a merge of that many vectors
   * @return what this writer prepared, for the caller to close, or {@code null} if it prepared
   *     nothing
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
   * vectors it merged.
   *
   * <p>It is returned from the merge of the flat vectors, {@link #scorerSupplier} may be called on
   * it at most once, by the caller that asked for it and once that caller has reopened the merged
   * segment, and the caller closes it either way: closing releases what the writer prepared unless
   * a supplier has taken it over. Not thread-safe.
   */
  public interface MergeScorerData extends Closeable {

    /**
     * Returns a scorer supplier over the merged vectors, or {@code null} if this data cannot serve
     * {@code mergedReader}, in which case the caller builds a scorer itself. This data is tied to
     * the segment its writer wrote, so a reader that wraps that segment's own flat reader is one it
     * cannot serve. The returned supplier takes over the prepared data and releases it when it is
     * closed; the caller must close it.
     *
     * <p>Called at most once, after the writer that prepared this has been finished and closed and
     * {@code mergedReader} has been opened over the segment it wrote. A second call, or a call
     * after {@link #close}, throws {@link IllegalStateException}.
     *
     * @param mergedReader a reader over the segment the writer wrote
     * @return a scorer supplier over the merged vectors, or {@code null} if this data cannot serve
     *     that reader
     * @throws IOException if an I/O error occurs
     */
    CloseableRandomVectorScorerSupplier scorerSupplier(FlatVectorsReader mergedReader)
        throws IOException;

    /**
     * Releases the prepared data unless {@link #scorerSupplier} has handed it to a supplier.
     * Idempotent, and best-effort: implementations must not throw, so that a caller can release on
     * the way out of a failed merge without replacing the exception that failed it.
     */
    @Override
    void close();
  }
}
