/**
 * The messages exchanged with the synthesis worker.
 *
 * Shared by both sides so a renamed field is a type error rather than a message
 * that is silently ignored at runtime.
 */

import { DownloadProgress } from './cache';
import type { Backend } from './repo';
import { SamplingOptions } from './types';
import { VoiceIndex } from './types';

export interface GenerateParams {
  segments: string[];
  /** A voice pack shipped with the weights, by name. Ignored when `voiceEmb`
   *  is set; empty for the unconditional prefix. */
  voiceName: string;
  /**
   * Latents supplied by the page instead of fetched by name — a pack the user
   * loaded off their own disk (see voicePack.ts).
   *
   * Passed through the message rather than kept in the worker: the page already
   * holds it to preview and label the voice, and a few tens of kilobytes per
   * generate is nothing next to a frame of audio.
   */
  voiceEmb?: Float32Array;
  options: Partial<SamplingOptions>;
  seed?: number;
}

export interface LoadedInfo {
  voices: VoiceIndex;
  base: string;
  sampleRate: number;
  /** The shape of the latents these weights condition on, so the page can
   *  reject a pack built for a different model before it is generated with —
   *  wrong latents are the right dtype and rank, and produce confident
   *  nonsense rather than an error. */
  nVoiceQueries: number;
  dModel: number;
  /** Which runtime loaded, so the page can say so. There is no silent fallback
   *  between the two: they read different model repositories, and quietly
   *  switching would turn a 206 MB download into an 858 MB one. */
  backend: Backend;
}

export type WorkerRequest =
  | { type: 'downloadInfo'; id: number; repo: string; backend?: Backend; gguf?: string }
  | { type: 'clearCache'; id: number }
  | { type: 'load'; id: number; repo: string; backend?: Backend; gguf?: string }
  | { type: 'generate'; id: number; params: GenerateParams }
  /** Cancels the in-flight `generate` whose id is `target`. */
  | { type: 'cancel'; id: number; target: number };

export type WorkerResponse =
  | { type: 'result'; id: number; value: unknown }
  | { type: 'error'; id: number; message: string }
  | { type: 'progress'; id: number; progress: DownloadProgress }
  | { type: 'chunk'; id: number; chunk: Float32Array };
