# Subtitle Playback Plugin

External subtitle playback plugin for Lator.

This plugin uses the executable runtime. Lator only downloads and launches the
assets declared in `plugin.json`; FFmpeg and FFplay are plugin resources rather
than built-in app playback features.

## Providers

- Background Video: cuts subtitle audio/video segments into the project cache
  and returns `lator-media://` URLs for inline playback.
- FFplay Window: opens the source media segment in a separate FFplay window.

The current asset URLs target Apple Silicon macOS builds.

Background video quality is controlled by this plugin's `videoSegmentMode`
setting, which is delivered in the executable request's `pluginSettings`.
