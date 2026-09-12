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

package org.apache.lucene.codecs.lucene95;

import java.io.IOException;
import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.SparseFixedBitSet;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.apache.lucene.util.packed.DirectMonotonicWriter;

/**
 * Configuration for {@link DirectMonotonicReader} and {@link IndexedDISI} for reading sparse
 * vectors. The format in the static writing methods adheres to the Lucene95HnswVectorsFormat
 */
public class OrdToDocDISIReaderConfiguration {

  /**
   * Writes out the docsWithField and ordToDoc mapping to the outputMeta and vectorData
   * respectively. This is in adherence to the Lucene95HnswVectorsFormat.
   *
   * <p>Within outputMeta the format is as follows:
   *
   * <ul>
   *   <li><b>[int8]</b> if equals to -2, empty - no vector values. If equals to -1, dense – all
   *       documents have values for a field. If equals to 0, sparse – some documents missing
   *       values.
   *   <li>DocIds were encoded by {@link IndexedDISI#writeBitSet(DocIdSetIterator, IndexOutput,
   *       byte)}
   *   <li>OrdToDoc was encoded by {@link org.apache.lucene.util.packed.DirectMonotonicWriter}, note
   *       that only in sparse case
   * </ul>
   *
   * <p>Within the vectorData the format is as follows:
   *
   * <ul>
   *   <li>DocIds encoded by {@link IndexedDISI#writeBitSet(DocIdSetIterator, IndexOutput, byte)},
   *       note that only in sparse case
   *   <li>OrdToDoc was encoded by {@link org.apache.lucene.util.packed.DirectMonotonicWriter}, note
   *       that only in sparse case
   * </ul>
   *
   * @param outputMeta the outputMeta
   * @param vectorData the vectorData
   * @param count the count of docs with vectors
   * @param maxDoc the maxDoc for the index
   * @param docsWithField the docs contaiting a vector field
   * @throws IOException thrown when writing data fails to either output
   */
  public static void writeStoredMeta(
      int directMonotonicBlockShift,
      IndexOutput outputMeta,
      IndexOutput vectorData,
      int count,
      int maxDoc,
      DocsWithFieldSet docsWithField)
      throws IOException {
    if (count == 0) {
      outputMeta.writeLong(-2); // docsWithFieldOffset
      outputMeta.writeLong(0L); // docsWithFieldLength
      outputMeta.writeShort((short) -1); // jumpTableEntryCount
      outputMeta.writeByte((byte) -1); // denseRankPower
    } else if (count == maxDoc) {
      outputMeta.writeLong(-1); // docsWithFieldOffset
      outputMeta.writeLong(0L); // docsWithFieldLength
      outputMeta.writeShort((short) -1); // jumpTableEntryCount
      outputMeta.writeByte((byte) -1); // denseRankPower
    } else {
      long offset = vectorData.getFilePointer();
      outputMeta.writeLong(offset); // docsWithFieldOffset
      final short jumpTableEntryCount =
          IndexedDISI.writeBitSet(
              docsWithField.iterator(), vectorData, IndexedDISI.DEFAULT_DENSE_RANK_POWER);
      outputMeta.writeLong(vectorData.getFilePointer() - offset); // docsWithFieldLength
      outputMeta.writeShort(jumpTableEntryCount);
      outputMeta.writeByte(IndexedDISI.DEFAULT_DENSE_RANK_POWER);

      // write ordToDoc mapping
      long start = vectorData.getFilePointer();
      outputMeta.writeLong(start);
      outputMeta.writeVInt(directMonotonicBlockShift);
      // dense case and empty case do not need to store ordToMap mapping
      final DirectMonotonicWriter ordToDocWriter =
          DirectMonotonicWriter.getInstance(
              outputMeta, vectorData, count, directMonotonicBlockShift);
      DocIdSetIterator iterator = docsWithField.iterator();
      for (int doc = iterator.nextDoc();
          doc != DocIdSetIterator.NO_MORE_DOCS;
          doc = iterator.nextDoc()) {
        ordToDocWriter.add(doc);
      }
      ordToDocWriter.finish();
      outputMeta.writeLong(vectorData.getFilePointer() - start);
    }
  }

  /**
   * Reads in the necessary fields stored in the outputMeta to configure {@link
   * DirectMonotonicReader} and {@link IndexedDISI}.
   *
   * @param inputMeta the inputMeta, previously written to via {@link #writeStoredMeta(int,
   *     IndexOutput, IndexOutput, int, int, DocsWithFieldSet)}
   * @param size The number of vectors
   * @return the configuration required to read sparse vectors
   * @throws IOException thrown when reading data fails
   */
  public static OrdToDocDISIReaderConfiguration fromStoredMeta(IndexInput inputMeta, int size)
      throws IOException {
    long docsWithFieldOffset = inputMeta.readLong();
    long docsWithFieldLength = inputMeta.readLong();
    short jumpTableEntryCount = inputMeta.readShort();
    byte denseRankPower = inputMeta.readByte();
    long addressesOffset = 0;
    int blockShift = 0;
    DirectMonotonicReader.Meta meta = null;
    long addressesLength = 0;
    if (docsWithFieldOffset > -1) {
      addressesOffset = inputMeta.readLong();
      blockShift = inputMeta.readVInt();
      meta = DirectMonotonicReader.loadMeta(inputMeta, size, blockShift);
      addressesLength = inputMeta.readLong();
    }
    return new OrdToDocDISIReaderConfiguration(
        size,
        jumpTableEntryCount,
        addressesOffset,
        addressesLength,
        docsWithFieldOffset,
        docsWithFieldLength,
        denseRankPower,
        meta);
  }

  final int size;
  // the following four variables used to read docIds encoded by IndexDISI
  // special values of docsWithFieldOffset are -1 and -2
  // -1 : dense
  // -2 : empty
  // other: sparse
  final short jumpTableEntryCount;
  final long docsWithFieldOffset, docsWithFieldLength;
  final byte denseRankPower;

  // the following four variables used to read ordToDoc encoded by DirectMonotonicWriter
  // note that only spare case needs to store ordToDoc
  final long addressesOffset, addressesLength;
  final DirectMonotonicReader.Meta meta;

  OrdToDocDISIReaderConfiguration(
      int size,
      short jumpTableEntryCount,
      long addressesOffset,
      long addressesLength,
      long docsWithFieldOffset,
      long docsWithFieldLength,
      byte denseRankPower,
      DirectMonotonicReader.Meta meta) {
    this.size = size;
    this.jumpTableEntryCount = jumpTableEntryCount;
    this.addressesOffset = addressesOffset;
    this.addressesLength = addressesLength;
    this.docsWithFieldOffset = docsWithFieldOffset;
    this.docsWithFieldLength = docsWithFieldLength;
    this.denseRankPower = denseRankPower;
    this.meta = meta;
  }

  /**
   * @param dataIn the dataIn
   * @return the IndexedDISI for sparse values
   * @throws IOException thrown when reading data fails
   */
  public IndexedDISI getIndexedDISI(IndexInput dataIn) throws IOException {
    assert docsWithFieldOffset > -1;
    return new IndexedDISI(
        dataIn,
        docsWithFieldOffset,
        docsWithFieldLength,
        jumpTableEntryCount,
        denseRankPower,
        size);
  }

  /**
   * How many word operations of a materialization one ordinal to doc lookup of the lazy accepted
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
   * The words of one DENSE block of an {@link IndexedDISI}, which is also what an ALL block costs.
   */
  private static final int WORDS_PER_BLOCK = 1 << 10;

  /**
   * A block of an {@link IndexedDISI} that holds fewer docs than this is SPARSE and costs one short
   * per doc to walk rather than a word per 64 docs, see {@code IndexedDISI#MAX_ARRAY_LENGTH}.
   */
  private static final int SPARSE_BLOCK_MAX_DOCS = 1 << 12;

  /**
   * Whether materializing the accepted ordinals is expected to cost less than {@code tests} lookups
   * through the lazy accepted ordinals, which is what it saves.
   *
   * <p>A materialization walks the blocks of the docs that have a vector, see {@link
   * IndexedDISI#indicesOf}: a DENSE or an ALL block costs {@code WORDS_PER_BLOCK} words, a SPARSE
   * block one short per doc, and a block is SPARSE when it holds fewer than {@code
   * SPARSE_BLOCK_MAX_DOCS} docs, so a block costs its docs when it holds fewer than that and {@code
   * WORDS_PER_BLOCK} words otherwise, taken as the average over the blocks: a nearly full SPARSE
   * block costs about four times a DENSE one. Every block may hold an accepted doc, so all of them
   * are charged, plus the bit set of {@code size} bits to zero for the result. Both sides are
   * counted in word operations. The cost does not depend on how many docs the filter accepts, so it
   * is the same for any filter over a given field, and the decision is about the tests alone.
   *
   * @param blocks the number of blocks of the docs that have a vector
   * @param size the number of docs that have a vector
   * @param tests the number of ordinals the caller expects to test if they are not materialized
   */
  public static boolean shouldMaterializeAcceptOrds(int blocks, int size, long tests) {
    assert blocks > 0 && size >= 0 && tests >= 0;
    long docsPerBlock = ((long) size + blocks - 1) / blocks;
    long wordsPerBlock = docsPerBlock < SPARSE_BLOCK_MAX_DOCS ? docsPerBlock : WORDS_PER_BLOCK;
    long words = blocks * wordsPerBlock + size / Long.SIZE;
    // tests * WORD_OPS_PER_LOOKUP >= words, without overflowing for a caller that tests everything
    return tests >= (words + WORD_OPS_PER_LOOKUP - 1) / WORD_OPS_PER_LOOKUP;
  }

  /**
   * Returns the ordinals of the vectors whose doc is set in {@code acceptDocs}, computed a word at
   * a time over the blocks of the docs that have a vector rather than a doc at a time, see {@link
   * IndexedDISI#indicesOf}, or {@code null} if {@code acceptDocs} is not a bit set that can be
   * walked that way, which is any {@link Bits} other than a {@link FixedBitSet} or a {@link
   * SparseFixedBitSet}, or if walking it is not expected to pay for {@code tests} lookups, see
   * {@link #shouldMaterializeAcceptOrds}. Only valid for the sparse configuration, the one that has
   * an {@link IndexedDISI}.
   *
   * @param dataIn the dataIn
   * @param acceptDocs the accepted docs
   * @param tests how many ordinals the caller expects to test if they are not materialized
   * @return a bit set of {@code size} bits over the ordinals, or {@code null}
   * @throws IOException thrown when reading data fails
   */
  public FixedBitSet materializeAcceptOrds(IndexInput dataIn, Bits acceptDocs, long tests)
      throws IOException {
    if ((acceptDocs instanceof FixedBitSet || acceptDocs instanceof SparseFixedBitSet) == false) {
      return null;
    }
    if (shouldMaterializeAcceptOrds(Math.max(1, jumpTableEntryCount), size, tests) == false) {
      return null;
    }
    FixedBitSet acceptOrds = new FixedBitSet(size);
    if (acceptDocs instanceof FixedBitSet fixedBitSet) {
      getIndexedDISI(dataIn).indicesOf(fixedBitSet, acceptOrds);
    } else {
      getIndexedDISI(dataIn).indicesOf((SparseFixedBitSet) acceptDocs, acceptOrds);
    }
    return acceptOrds;
  }

  /**
   * @param dataIn the dataIn
   * @return the DirectMonotonicReader for sparse values
   * @throws IOException thrown when reading data fails
   */
  public DirectMonotonicReader getDirectMonotonicReader(IndexInput dataIn) throws IOException {
    assert docsWithFieldOffset > -1;
    final RandomAccessInput addressesData =
        dataIn.randomAccessSlice(addressesOffset, addressesLength);
    return DirectMonotonicReader.getInstance(meta, addressesData);
  }

  /**
   * @return If true, the field is empty, no vector values. If false, the field is either dense or
   *     sparse.
   */
  public boolean isEmpty() {
    return docsWithFieldOffset == -2;
  }

  /**
   * @return If true, the field is dense, all documents have values for a field. If false, the field
   *     is sparse, some documents missing values.
   */
  public boolean isDense() {
    return docsWithFieldOffset == -1;
  }
}
