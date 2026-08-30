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
package org.apache.lucene.index;

import java.io.IOException;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

/**
 * This class abstracts addressing of document vector values indexed as {@link KnnFloatVectorField}
 * or {@link KnnByteVectorField}.
 *
 * @lucene.experimental
 */
public abstract class KnnVectorValues {

  /** Return the dimension of the vectors */
  public abstract int dimension();

  /**
   * Return the number of vectors for this field.
   *
   * @return the number of vectors returned by this iterator
   */
  public abstract int size();

  /**
   * Return the docid of the document indexed with the given vector ordinal. This default
   * implementation returns the argument and is appropriate for dense values implementations where
   * every doc has a single value.
   */
  public int ordToDoc(int ord) {
    return ord;
  }

  /**
   * Prefetches the provided ordinals
   *
   * @param ordsToPrefetch a list of ordinals to prefetch
   * @param numOrds number of ords to prefetch from ordsToPrefetch array
   */
  public void prefetch(final int[] ordsToPrefetch, int numOrds) throws IOException {}

  /**
   * Creates a new copy of this {@link KnnVectorValues}. This is helpful when you need to access
   * different values at once, to avoid overwriting the underlying vector returned.
   */
  public abstract KnnVectorValues copy() throws IOException;

  /** Returns the vector byte length, defaults to dimension multiplied by float byte size */
  public int getVectorByteLength() {
    return dimension() * getEncoding().byteSize;
  }

  /** The vector encoding of these values. */
  public abstract VectorEncoding getEncoding();

  /**
   * Returns a Bits accepting docs accepted by the argument and having a vector value.
   *
   * <p>The bits are indexed by ordinal rather than by doc. An implementation that returns {@code
   * acceptDocs} itself is asserting that its ordinals are its docs, which is the invariant that
   * {@link #ordToDoc(int)} returns the argument, and that dense values rely on to enumerate their
   * accepted ordinals in {@link #acceptedOrdsIterator(Bits, DocIdSetIterator)}.
   */
  public Bits getAcceptOrds(Bits acceptDocs) {
    // FIXME: change default to return acceptDocs and provide this impl
    // somewhere more specialized (in every non-dense impl).
    if (acceptDocs == null) {
      return null;
    }
    return new Bits() {
      @Override
      public boolean get(int index) {
        return acceptDocs.get(ordToDoc(index));
      }

      @Override
      public int length() {
        return size();
      }
    };
  }

  /**
   * Returns a Bits accepting docs accepted by the argument and having a vector value, like {@link
   * #getAcceptOrds(Bits)}, materialized up-front rather than mapping every tested ordinal to its
   * doc.
   *
   * <p>Implementations that map ordinals to docs through an off-heap structure answer the same
   * question more cheaply by walking {@code acceptDocsIterator} and the docs that have a vector
   * once, as long as the caller tests more ordinals than the filter accepts docs. The default
   * implementation ignores the iterator and returns the lazy view, which is what dense values want:
   * their ordinals are their docs, so there is nothing to materialize.
   *
   * <p>An implementation that materializes them returns a {@link BitSet}, so that a caller that
   * would rather enumerate the accepted ordinals than test them one at a time can do so through
   * {@link #acceptedOrdsIterator(Bits, DocIdSetIterator)}.
   *
   * @param acceptDocs the accepted docs, or {@code null} if all docs are accepted
   * @param acceptDocsIterator an iterator over the same docs as {@code acceptDocs}, which
   *     implementations are free to consume entirely
   */
  public Bits getAcceptOrds(Bits acceptDocs, DocIdSetIterator acceptDocsIterator)
      throws IOException {
    return getAcceptOrds(acceptDocs);
  }

  /**
   * Returns an iterator over the ordinals of the vectors whose doc is accepted, in increasing
   * order, or {@code null} when they cannot be enumerated and a caller has to test every ordinal
   * against {@link #getAcceptOrds(Bits)} instead.
   *
   * <p>The iterator visits exactly the ordinals that {@link #getAcceptOrds(Bits)} accepts, so a
   * caller that scores all of them scores every accepted vector once and nothing else.
   *
   * <p>The default implementation materializes them through {@link #getAcceptOrds(Bits,
   * DocIdSetIterator)}, which answers with a {@link BitSet} in the implementations that map
   * ordinals to docs through an off-heap structure, and with the lazy view in the ones that do not,
   * which cannot be enumerated. Dense values override this to hand back the accepted docs
   * themselves, since their ordinals are their docs.
   *
   * @param acceptDocs the accepted docs, or {@code null} if all docs are accepted
   * @param acceptDocsIterator an iterator over the same docs as {@code acceptDocs}, which
   *     implementations are free to consume entirely
   */
  public DocIdSetIterator acceptedOrdsIterator(Bits acceptDocs, DocIdSetIterator acceptDocsIterator)
      throws IOException {
    if (acceptDocs == null) {
      return null;
    }
    if (getAcceptOrds(acceptDocs, acceptDocsIterator) instanceof BitSet acceptedOrds) {
      assert acceptedOrds.length() == size()
          : "accepted ordinals of " + acceptedOrds.length() + " bits for " + size() + " vectors";
      return new BitSetIterator(acceptedOrds, acceptedOrds.approximateCardinality());
    }
    return null;
  }

  /**
   * Materializes into a bit set the ordinals of the vectors whose doc is accepted, by leap frogging
   * the accepted docs with the docs that have a vector.
   *
   * <p>Both iterators must be positioned before their first doc, and {@code vectors} must keep
   * {@link DocIndexIterator#index()} in sync with its doc after an {@link
   * DocIndexIterator#advance(int)}, which is the case for iterators over a disk-based doc id set
   * but not for {@link #fromDISI}, whose ordinal only moves on {@link DocIndexIterator#nextDoc()}.
   * The assertions below check that invariant.
   *
   * <p>The bit set is returned rather than a read-only view of it, so that a caller that would
   * rather enumerate the accepted ordinals than test them one at a time can do so. Nothing else
   * holds a reference to it.
   *
   * @param acceptDocsIterator an iterator over the accepted docs, which this method consumes
   * @param vectors an iterator over the docs that have a vector, which this method consumes
   */
  protected final FixedBitSet materializeAcceptOrds(
      DocIdSetIterator acceptDocsIterator, DocIndexIterator vectors) throws IOException {
    FixedBitSet acceptedOrds = new FixedBitSet(size());
    int acceptedDoc = acceptDocsIterator.nextDoc();
    int vectorDoc = vectors.nextDoc();
    int previousOrd = -1;
    while (acceptedDoc != DocIdSetIterator.NO_MORE_DOCS
        && vectorDoc != DocIdSetIterator.NO_MORE_DOCS) {
      if (vectorDoc < acceptedDoc) {
        vectorDoc = vectors.advance(acceptedDoc);
      } else if (vectorDoc == acceptedDoc) {
        assert vectors.docID() == acceptedDoc;
        int ord = vectors.index();
        assert ord > previousOrd
            : "ordinals must increase with docs: " + ord + " <= " + previousOrd;
        previousOrd = ord;
        acceptedOrds.set(ord);
        acceptedDoc = acceptDocsIterator.nextDoc();
      } else {
        acceptedDoc = acceptDocsIterator.advance(vectorDoc);
      }
    }
    return acceptedOrds;
  }

  /** Create an iterator for this instance. */
  public DocIndexIterator iterator() {
    throw new UnsupportedOperationException();
  }

  /**
   * A DocIdSetIterator that also provides an index() method tracking a distinct ordinal for a
   * vector associated with each doc.
   */
  public abstract static class DocIndexIterator extends DocIdSetIterator {

    /** return the value index (aka "ordinal" or "ord") corresponding to the current doc */
    public abstract int index();
  }

  /**
   * Creates an iterator for instances where every doc has a value, and the value ordinals are equal
   * to the docids.
   */
  protected DocIndexIterator createDenseIterator() {
    return new DocIndexIterator() {

      int doc = -1;

      @Override
      public int docID() {
        return doc;
      }

      @Override
      public int index() {
        return doc;
      }

      @Override
      public int nextDoc() throws IOException {
        if (doc >= size() - 1) {
          return doc = NO_MORE_DOCS;
        } else {
          return ++doc;
        }
      }

      @Override
      public int advance(int target) {
        if (target >= size()) {
          return doc = NO_MORE_DOCS;
        }
        return doc = target;
      }

      @Override
      public long cost() {
        return size();
      }
    };
  }

  /**
   * Creates an iterator from a DocIdSetIterator indicating which docs have values, and for which
   * ordinals increase monotonically with docid.
   */
  protected static DocIndexIterator fromDISI(DocIdSetIterator docsWithField) {
    return new DocIndexIterator() {

      int ord = -1;

      @Override
      public int docID() {
        return docsWithField.docID();
      }

      @Override
      public int index() {
        return ord;
      }

      @Override
      public int nextDoc() throws IOException {
        if (docID() == NO_MORE_DOCS) {
          return NO_MORE_DOCS;
        }
        ord++;
        return docsWithField.nextDoc();
      }

      @Override
      public int advance(int target) throws IOException {
        return docsWithField.advance(target);
      }

      @Override
      public long cost() {
        return docsWithField.cost();
      }
    };
  }

  /**
   * Creates an iterator from this instance's ordinal-to-docid mapping which must be monotonic
   * (docid increases when ordinal does).
   */
  protected DocIndexIterator createSparseIterator() {
    return new DocIndexIterator() {
      private int ord = -1;

      @Override
      public int docID() {
        if (ord == -1) {
          return -1;
        }
        if (ord == NO_MORE_DOCS) {
          return NO_MORE_DOCS;
        }
        return ordToDoc(ord);
      }

      @Override
      public int index() {
        return ord;
      }

      @Override
      public int nextDoc() throws IOException {
        if (ord >= size() - 1) {
          ord = NO_MORE_DOCS;
        } else {
          ++ord;
        }
        return docID();
      }

      @Override
      public int advance(int target) throws IOException {
        return slowAdvance(target);
      }

      @Override
      public long cost() {
        return size();
      }
    };
  }
}
