/**
 * Reading a voice pack out of the zip the user dropped on the page.
 *
 * A voice downloaded from zeroweight.ai is a zip holding one folder —
 * `voice.npz` (required), `voice.bin`, `preview.wav`, `meta.json`, the layout
 * docs/VOICES.md describes. The file goes in as it came down: no unzipping, no
 * folder picking, and nothing is uploaded — it is read in this tab and the
 * latents go to the worker.
 *
 * So there are two ZIP layers here, the pack archive and the `.npz` inside it
 * (which is itself a zip of `.npy` members), and one small reader serves both.
 * `voice.bin` is preferred over `voice.npz` when a pack carries it, because it
 * is the raw float32 array and needs no parsing at all.
 */

/** One pack, ready for the picker. */
export interface LocalVoice {
  /** Directory name inside the zip — how the voice is addressed, as in the
   *  Python package. */
  name: string;
  displayName: string;
  language?: string;
  description?: string;
  tags: string[];
  /** The latents, flat: (1, n_voice_queries, d_model) in C order. */
  emb: Float32Array;
  /** The preview clip, kept as a blob so it can be stored and replayed. */
  preview: Blob | null;
}

export interface ReadResult {
  voices: LocalVoice[];
  /** One line per pack that was found and could not be read. The caller shows
   *  them: a zip of five voices with one truncated should install four and say
   *  which one it dropped. */
  errors: string[];
}

/** The files a pack is made of. Everything else in the archive is ignored — a
 *  download carries a README, and a zip that has been through a Mac carries
 *  `__MACOSX/` and `.DS_Store` noise. */
const PACK_FILES = ['voice.npz', 'voice.bin', 'preview.wav', 'meta.json'] as const;
type PackFile = (typeof PACK_FILES)[number];

// ── zip ──────────────────────────────────────────────────────────────────────

interface ZipMember {
  name: string;
  method: number;
  compressedSize: number;
  offset: number;
}

/** Central-directory listing. No ZIP64 — a voice pack is tens of kilobytes. */
function zipMembers(buffer: ArrayBuffer): ZipMember[] {
  const view = new DataView(buffer);
  // The end-of-central-directory record is last, but a comment may follow it,
  // so scan back rather than assuming it sits at exactly -22.
  let eocd = -1;
  for (let i = buffer.byteLength - 22; i >= 0 && i > buffer.byteLength - 66_000; i--) {
    if (view.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error('không phải file zip hợp lệ');

  const count = view.getUint16(eocd + 10, true);
  let p = view.getUint32(eocd + 16, true);
  const members: ZipMember[] = [];
  for (let i = 0; i < count; i++) {
    if (view.getUint32(p, true) !== 0x02014b50) throw new Error('zip bị hỏng');
    const nameLen = view.getUint16(p + 28, true);
    const extraLen = view.getUint16(p + 30, true);
    const commentLen = view.getUint16(p + 32, true);
    members.push({
      name: new TextDecoder().decode(new Uint8Array(buffer, p + 46, nameLen)),
      method: view.getUint16(p + 10, true),
      compressedSize: view.getUint32(p + 20, true),
      offset: view.getUint32(p + 42, true),
    });
    p += 46 + nameLen + extraLen + commentLen;
  }
  return members;
}

async function readMember(buffer: ArrayBuffer, member: ZipMember): Promise<ArrayBuffer> {
  const view = new DataView(buffer);
  if (view.getUint32(member.offset, true) !== 0x04034b50) {
    throw new Error(`mục ${member.name} trong zip bị hỏng`);
  }
  // The local header's own name/extra lengths, not the central directory's:
  // the extra field is routinely a different size in the two.
  const start = member.offset + 30
    + view.getUint16(member.offset + 26, true)
    + view.getUint16(member.offset + 28, true);
  const raw = buffer.slice(start, start + member.compressedSize);

  if (member.method === 0) return raw;
  if (member.method !== 8) throw new Error(`zip dùng kiểu nén không hỗ trợ (${member.method})`);
  // The platform's download stores; a zip re-made by a file manager deflates.
  // `np.savez` stores, `np.savez_compressed` deflates. All four are valid.
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('trình duyệt này không giải nén được zip nén');
  }
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Response(stream).arrayBuffer();
}

// ── .npy ─────────────────────────────────────────────────────────────────────

function parseNpyFloat32(buffer: ArrayBuffer, what: string): Float32Array {
  const bytes = new Uint8Array(buffer);
  if (String.fromCharCode(...bytes.subarray(1, 6)) !== 'NUMPY') {
    throw new Error(`${what}: không phải file .npy`);
  }
  const view = new DataView(buffer);
  const major = bytes[6];
  const headerLen = major >= 2 ? view.getUint32(8, true) : view.getUint16(8, true);
  const start = major >= 2 ? 12 : 10;
  const header = new TextDecoder().decode(bytes.subarray(start, start + headerLen));

  if (!/'descr':\s*'[<|]f4'/.test(header)) {
    throw new Error(`${what}: voice_emb phải là float32`);
  }
  if (/'fortran_order':\s*True/.test(header)) {
    throw new Error(`${what}: không hỗ trợ Fortran order`);
  }
  // Copied rather than viewed: NumPy pads its header to a 64-byte boundary so
  // the payload is normally aligned, but a re-saved file need not be, and a
  // Float32Array view of an unaligned offset throws.
  return new Float32Array(buffer.slice(start + headerLen));
}

/** The `voice_emb` array out of a `voice.npz`. */
async function embFromNpz(buffer: ArrayBuffer): Promise<Float32Array> {
  const member = zipMembers(buffer).find((m) => m.name.replace(/\.npy$/, '') === 'voice_emb');
  if (!member) throw new Error('voice.npz không có mảng voice_emb');
  return parseNpyFloat32(await readMember(buffer, member), 'voice.npz');
}

// ── packs ────────────────────────────────────────────────────────────────────

/**
 * Pack name -> its files, for every pack in the archive.
 *
 * A pack is a folder holding a `voice.npz`. The download ships exactly one,
 * under its own name; a zip of several is the obvious thing for someone to
 * build and costs nothing to accept. A zip whose ROOT is the pack takes its
 * name from the file, since there is no folder to take one from.
 */
function groupPacks(members: ZipMember[], fallbackName: string) {
  const packs = new Map<string, Map<PackFile, ZipMember>>();
  for (const member of members) {
    const parts = member.name.replace(/\\/g, '/').split('/').filter(Boolean);
    // Mac Finder writes a parallel __MACOSX/ tree of resource forks whose paths
    // mirror the real ones — following it would "find" every pack twice, the
    // second time with unreadable files.
    if (!parts.length || parts[0] === '__MACOSX') continue;
    const base = parts[parts.length - 1] as PackFile;
    if (!PACK_FILES.includes(base)) continue;

    const name = parts.length > 1 ? parts[parts.length - 2] : fallbackName;
    let files = packs.get(name);
    if (!files) packs.set(name, (files = new Map()));
    files.set(base, member);
  }
  for (const [name, files] of packs) {
    if (!files.has('voice.npz') && !files.has('voice.bin')) packs.delete(name);
  }
  return packs;
}

const str = (value: unknown) => (typeof value === 'string' ? value : undefined);

async function readPack(
  buffer: ArrayBuffer, name: string, files: Map<PackFile, ZipMember>,
): Promise<LocalVoice> {
  const bin = files.get('voice.bin');
  const npz = files.get('voice.npz');
  const emb = bin
    ? new Float32Array(await readMember(buffer, bin))
    : await embFromNpz(await readMember(buffer, npz as ZipMember));
  if (!emb.length) throw new Error('latents rỗng');

  // A pack with unreadable metadata is still a usable voice — it just shows
  // under its folder name.
  let meta: Record<string, unknown> = {};
  const metaMember = files.get('meta.json');
  if (metaMember) {
    try {
      meta = JSON.parse(new TextDecoder().decode(await readMember(buffer, metaMember)));
    } catch { /* keep the defaults */ }
  }

  const previewMember = files.get('preview.wav');
  const preview = previewMember
    ? new Blob([await readMember(buffer, previewMember)], { type: 'audio/wav' })
    : null;

  return {
    name,
    displayName: str(meta.display_name) || name,
    language: str(meta.language),
    description: str(meta.description),
    tags: Array.isArray(meta.tags)
      ? meta.tags.filter((t): t is string => typeof t === 'string') : [],
    emb,
    preview,
  };
}

/**
 * Every voice in a downloaded zip.
 *
 * Never rejects on one bad pack: the failures come back in `errors` beside the
 * voices that did load.
 */
export async function readVoiceZip(file: File): Promise<ReadResult> {
  const buffer = await file.arrayBuffer();
  const fallback = file.name.replace(/\.zip$/i, '') || 'custom-voice';

  let packs: ReturnType<typeof groupPacks>;
  try {
    packs = groupPacks(zipMembers(buffer), fallback);
  } catch (error) {
    return { voices: [], errors: [`${file.name}: ${(error as Error).message}`] };
  }
  if (!packs.size) {
    return {
      voices: [],
      errors: [`${file.name} không chứa giọng nào — cần một thư mục có voice.npz bên trong.`],
    };
  }

  const voices: LocalVoice[] = [];
  const errors: string[] = [];
  for (const [name, files] of packs) {
    try {
      voices.push(await readPack(buffer, name, files));
    } catch (error) {
      errors.push(`${name}: ${(error as Error).message}`);
    }
  }
  voices.sort((a, b) => a.name.localeCompare(b.name));
  return { voices, errors };
}
