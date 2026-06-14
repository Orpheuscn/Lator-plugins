#!/usr/bin/env node

import { spawn } from 'node:child_process'
import { stat } from 'node:fs/promises'
import { join } from 'node:path'
import process from 'node:process'

const FFPLAY_CAPABILITY_ID = 'subtitle-playback.ffplay-window'
const BACKGROUND_VIDEO_CAPABILITY_ID = 'subtitle-playback.background-video'
const FFPLAY_ASSET_ID = 'ffplay-darwin-arm64'
const FFMPEG_ASSET_ID = 'ffmpeg-darwin-arm64'

let activeChild = null

process.on('SIGTERM', () => {
  if (activeChild && !activeChild.killed) {
    activeChild.kill('SIGTERM')
  }
  process.exit(143)
})

process.on('SIGINT', () => {
  if (activeChild && !activeChild.killed) {
    activeChild.kill('SIGTERM')
  }
  process.exit(130)
})

try {
  const request = JSON.parse(await readStdin())
  const params = isRecord(request.params) ? request.params : {}

  if (request.capabilityId === FFPLAY_CAPABILITY_ID) {
    const result = await playWithFfplay(request, params)
    writeStructuredLine({ result })
  } else if (request.capabilityId === BACKGROUND_VIDEO_CAPABILITY_ID) {
    const result = await playWithBackgroundVideo(request, params)
    writeStructuredLine({ result })
  } else {
    throw new Error(`Unknown subtitle playback capability: ${request.capabilityId || 'unknown'}`)
  }
} catch (error) {
  writeStructuredLine({
    error: error instanceof Error ? error.message : String(error)
  })
  process.exitCode = 1
}

async function playWithFfplay(request, params) {
  const sourcePath = getString(params.hostMedia, 'sourceVideoPath') ||
    getString(params.hostMedia, 'sourceAudioPath')
  if (!sourcePath) {
    throw new Error('No source media path is available for FFplay playback.')
  }

  const ffplayPath = await requireAssetExecutable(request, FFPLAY_ASSET_ID, 'ffplay')
  const startSeconds = getRequiredNumber(params, 'startSeconds')
  const durationSeconds = getRequiredNumber(params.hostMedia, 'durationSeconds')
  const segmentId = getRequiredString(params, 'segmentId')

  await runProcess(ffplayPath, [
    '-hide_banner',
    '-autoexit',
    '-alwaysontop',
    '-ss',
    startSeconds.toFixed(3),
    '-t',
    durationSeconds.toFixed(3),
    '-window_title',
    `Lator subtitle ${segmentId}`,
    sourcePath
  ], {
    env: {
      ...process.env,
      ...(process.platform === 'darwin' ? { SDL_MAC_BACKGROUND_APP: '1' } : {})
    }
  })

  writeStructuredLine({ event: { type: 'ended' } })
  return { mode: 'external-window' }
}

async function playWithBackgroundVideo(request, params) {
  const ffmpegPath = await requireAssetExecutable(request, FFMPEG_ASSET_ID, 'ffmpeg')
  const hostMedia = isRecord(params.hostMedia) ? params.hostMedia : {}
  const pluginSettings = isRecord(request.pluginSettings) ? request.pluginSettings : {}
  const startSeconds = getRequiredNumber(params, 'startSeconds')
  const durationSeconds = getRequiredNumber(hostMedia, 'durationSeconds')
  const videoMode = getString(pluginSettings, 'videoSegmentMode') === 'compressed' ? 'compressed' : 'source'
  const result = {}

  const sourceAudioPath = getString(hostMedia, 'sourceAudioPath') || getString(hostMedia, 'sourceVideoPath')
  const inlineAudioOutputPath = getString(hostMedia, 'inlineAudioOutputPath')
  const inlineAudioUrl = getString(hostMedia, 'inlineAudioUrl')
  if (sourceAudioPath && inlineAudioOutputPath && inlineAudioUrl) {
    if (!(await isReadyFile(inlineAudioOutputPath))) {
      writeStructuredLine({ event: { type: 'progress', stage: 'extracting-audio' } })
      await runProcess(ffmpegPath, buildAudioSegmentArgs(
        sourceAudioPath,
        inlineAudioOutputPath,
        startSeconds,
        durationSeconds
      ))
    }
    result.audioUrl = inlineAudioUrl
  }

  const sourceVideoPath = getString(hostMedia, 'sourceVideoPath')
  const inlineVideoOutputPath = getString(hostMedia, 'inlineVideoOutputPath')
  const inlineVideoUrl = getString(hostMedia, 'inlineVideoUrl')
  if (sourceVideoPath && inlineVideoOutputPath && inlineVideoUrl) {
    if (!(await isReadyFile(inlineVideoOutputPath))) {
      writeStructuredLine({ event: { type: 'progress', stage: 'extracting-video' } })
      await runProcess(ffmpegPath, buildVideoSegmentArgs(
        sourceVideoPath,
        inlineVideoOutputPath,
        startSeconds,
        durationSeconds,
        videoMode
      ))
    }
    result.videoUrl = inlineVideoUrl
  }

  if (!result.audioUrl && !result.videoUrl) {
    throw new Error('No inline subtitle playback output is available.')
  }

  writeStructuredLine({ event: { type: 'done', ...result } })
  return {
    mode: 'background-video',
    ...result
  }
}

function buildAudioSegmentArgs(inputPath, outputPath, startSeconds, durationSeconds) {
  return [
    '-hide_banner',
    '-y',
    '-ss',
    startSeconds.toFixed(3),
    '-i',
    inputPath,
    '-t',
    durationSeconds.toFixed(3),
    '-map',
    '0:a:0',
    '-vn',
    '-c:a',
    'aac',
    '-b:a',
    '48k',
    '-ac',
    '1',
    '-movflags',
    '+faststart',
    outputPath
  ]
}

function buildVideoSegmentArgs(inputPath, outputPath, startSeconds, durationSeconds, mode) {
  return [
    '-hide_banner',
    '-y',
    '-ss',
    startSeconds.toFixed(3),
    '-i',
    inputPath,
    '-t',
    durationSeconds.toFixed(3),
    '-map',
    '0:v:0',
    '-c:v',
    'libx264',
    '-preset',
    mode === 'compressed' ? 'veryfast' : 'fast',
    '-crf',
    mode === 'compressed' ? '28' : '20',
    '-pix_fmt',
    'yuv420p',
    '-an',
    '-map_metadata',
    '-1',
    '-map_chapters',
    '-1',
    '-movflags',
    '+faststart',
    outputPath
  ]
}

async function runProcess(executablePath, args, options = {}) {
  await new Promise((resolve, reject) => {
    let stderr = ''
    const child = spawn(executablePath, args, {
      env: options.env || process.env,
      stdio: ['ignore', 'ignore', 'pipe']
    })
    activeChild = child

    child.stderr.setEncoding('utf8')
    child.stderr.on('data', chunk => {
      stderr += chunk
      if (stderr.length > 12000) {
        stderr = stderr.slice(-12000)
      }
    })
    child.on('error', error => {
      if (activeChild === child) activeChild = null
      reject(error)
    })
    child.on('close', code => {
      if (activeChild === child) activeChild = null
      if (code === 0) {
        resolve()
        return
      }

      reject(new Error(stderr.trim() || `${executablePath} exited with code ${code ?? 'unknown'}.`))
    })
  })
}

async function requireAssetExecutable(request, assetId, executableName) {
  const assets = isRecord(request.pluginAssetPaths) ? request.pluginAssetPaths : {}
  const assetPath = getString(assets, assetId)
  if (!assetPath) {
    throw new Error(`Missing plugin asset path: ${assetId}`)
  }

  const executablePath = join(assetPath, executableName)
  const fileStat = await stat(executablePath)
  if (!fileStat.isFile()) {
    throw new Error(`Plugin asset executable is not a file: ${executableName}`)
  }
  return executablePath
}

async function isReadyFile(filePath) {
  try {
    const fileStat = await stat(filePath)
    return fileStat.isFile() && fileStat.size > 0
  } catch {
    return false
  }
}

function readStdin() {
  return new Promise((resolve, reject) => {
    let input = ''
    process.stdin.setEncoding('utf8')
    process.stdin.on('data', chunk => {
      input += chunk
    })
    process.stdin.on('error', reject)
    process.stdin.on('end', () => resolve(input.trim()))
  })
}

function writeStructuredLine(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`)
}

function getRequiredString(value, key) {
  const item = getString(value, key)
  if (!item) {
    throw new Error(`Missing required field: ${key}`)
  }
  return item
}

function getString(value, key) {
  if (!isRecord(value)) return ''
  const item = value[key]
  return typeof item === 'string' && item.trim() ? item.trim() : ''
}

function getRequiredNumber(value, key) {
  if (!isRecord(value) || typeof value[key] !== 'number' || !Number.isFinite(value[key])) {
    throw new Error(`Missing required number: ${key}`)
  }
  return value[key]
}

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}
