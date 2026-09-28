/**
 * Where a dropped voice is kept, so dropping it is a one-time act.
 *
 * The Python side copies an installed pack into `~/.zerotts/voices`; this is
 * the browser's equivalent. A voice survives a reload and a closed tab, and is
 * in the picker the next time the page opens — which is the difference between
 * "load your voice every visit" and "your voices".
 *
 * IndexedDB rather than localStorage because the payload is binary and megabyte-
 * ish: localStorage is strings only, and base64 would cost a third more for the
 * privilege of a much smaller quota. The model itself lives in the Cache API for
 * the same reason.
 *
 * Every operation degrades to a no-op rather than throwing. A private window, a
 * browser with storage disabled, or a full quota should cost persistence and
 * nothing else — the voice still works for this session.
 */

import type { LocalVoice } from './voicePack';

const DB_NAME = 'zerotts-voices';
const STORE = 'voices';
const VERSION = 1;

let dbPromise: Promise<IDBDatabase | null> | null = null;

function openDb(): Promise<IDBDatabase | null> {
  dbPromise ??= new Promise<IDBDatabase | null>((resolve) => {
    if (typeof indexedDB === 'undefined') { resolve(null); return; }
    let request: IDBOpenDBRequest;
    try {
      request = indexedDB.open(DB_NAME, VERSION);
    } catch {
      resolve(null);
      return;
    }
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE)) {
        request.result.createObjectStore(STORE, { keyPath: 'name' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    // Blocked (another tab holds an older version) resolves too: waiting
    // forever would hang the page on something the user cannot see.
    request.onerror = () => resolve(null);
    request.onblocked = () => resolve(null);
  });
  return dbPromise;
}

function run<T>(
  mode: IDBTransactionMode,
  action: (store: IDBObjectStore) => IDBRequest<T>,
  fallback: T,
): Promise<T> {
  return openDb().then((db) => {
    if (!db) return fallback;
    return new Promise<T>((resolve) => {
      let request: IDBRequest<T>;
      try {
        request = action(db.transaction(STORE, mode).objectStore(STORE));
      } catch {
        resolve(fallback);
        return;
      }
      request.onsuccess = () => resolve(request.result ?? fallback);
      request.onerror = () => resolve(fallback);
    });
  });
}

/**
 * What actually goes in the store.
 *
 * A plain object rather than the `LocalVoice` itself: structured clone handles
 * both, but writing the shape down is what keeps a future field on `LocalVoice`
 * (a preview object URL, say — dead the moment the page closes) from being
 * silently persisted.
 */
interface StoredVoice {
  name: string;
  displayName: string;
  language?: string;
  description?: string;
  tags: string[];
  emb: Float32Array;
  preview: Blob | null;
}

export async function saveVoice(voice: LocalVoice): Promise<void> {
  const record: StoredVoice = {
    name: voice.name,
    displayName: voice.displayName,
    language: voice.language,
    description: voice.description,
    tags: voice.tags,
    emb: voice.emb,
    preview: voice.preview,
  };
  await run('readwrite', (store) => store.put(record) as IDBRequest<unknown>, null);
}

/** Every installed voice, oldest key first. Empty when storage is unavailable. */
export async function loadVoices(): Promise<LocalVoice[]> {
  const records = await run<StoredVoice[]>(
    'readonly', (store) => store.getAll() as IDBRequest<StoredVoice[]>, []);
  // A record written by a future version could be missing what the picker
  // needs; drop it rather than render a voice with no latents.
  return records
    .filter((r) => r && r.name && r.emb instanceof Float32Array && r.emb.length > 0)
    .map((r) => ({ ...r, tags: r.tags ?? [], preview: r.preview ?? null }));
}

export async function deleteVoice(name: string): Promise<void> {
  await run('readwrite', (store) => store.delete(name) as IDBRequest<unknown>, null);
}
