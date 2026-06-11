# Input Guide

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=700&size=28&duration=2500&pause=800&color=06b6d4&center=true&vCenter=true&width=500&lines=Input+Guide;Getting+the+Best+Results;Audio+Recording+Tips;Format+Compatibility" alt="Typing Animation" />
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=06b6d4,6366f1&height=100&text=Input+Guide&fontSize=30&fontColor=fff" alt="Banner" />
</p>

---

## Supported Audio Formats

| Format | Extension | Quality | Recommended |
|--------|-----------|---------|-------------|
| WAV | `.wav` | Lossless | Yes |
| FLAC | `.flac` | Lossless | Yes |
| MP3 | `.mp3` | Compressed | Yes |
| OGG | `.ogg` | Compressed | Yes |
| M4A | `.m4a` | Compressed | Yes |
| WEBM | `.webm` | Web format | Yes |

---

## Recording Best Practices

### 1. Environment Setup
- Record in a **quiet room** with minimal background noise
- Turn off fans, air conditioners, and music
- Use soft furnishings to reduce echo (curtains, cushions)
- Avoid outdoor recording when possible

### 2. Microphone Position

| Distance | Result |
|----------|--------|
| 0-5 cm | Too close - clipping/distortion |
| 6-12 cm | Optimal - clear signal |
| 13-30 cm | Acceptable - may lose subtle cues |
| 30+ cm | Too far - low signal-to-noise ratio |

- Position microphone at **mouth level**
- Speak across the mic, not directly into it
- Use a pop filter if available

### 3. Speaking Guidelines
- Speak in your **natural voice** - do not exaggerate emotions
- Maintain **consistent volume** throughout
- Avoid whispering or shouting
- Speak for at least **3 seconds** (system analyzes first 3s)
- Use short phrases or sentences for best results

---

## Audio Specifications

```yaml
sample_rate: 22050 Hz or higher
channels: Mono (preferred) or Stereo
bit_depth: 16-bit or 24-bit
max_duration: No limit (first 3s analyzed)
max_file_size: 32 MB
```

---

## Common Mistakes to Avoid

| Mistake | Why It Hurts | Solution |
|---------|--------------|----------|
| Background music | Confuses spectral features | Remove all music |
| Multiple speakers | Mixed emotional signals | One speaker per file |
| Heavy compression | Loses emotional nuance | Use WAV/FLAC when possible |
| Silence at start | Wastes analysis window | Trim leading silence |
| Phone call audio | Low bandwidth artifacts | Use direct recording |

---

## Understanding Your Results

### Confidence Score Guide

| Score | Meaning | Action |
|-------|---------|--------|
| 85-100% | High confidence | Result is highly reliable |
| 70-84% | Good confidence | Likely correct, minor ambiguity |
| 50-69% | Moderate confidence | Mixed emotions possible |
| Below 50% | Low confidence | Check audio quality, re-record |

### Top 3 Emotions

The dashboard shows the **3 most likely emotions** ranked by probability:
- **#1** - Primary detected emotion
- **#2** - Secondary emotional undertone
- **#3** - Tertiary influence

If #2 and #3 are close to #1, the speaker may be experiencing **mixed emotions**.

---

## Troubleshooting

### Upload Issues
**Problem:** File upload fails  
**Solutions:**
- Check file size is under 32MB
- Verify file extension is supported
- Try converting to WAV format
- Clear browser cache and retry

### Low Confidence Scores
**Problem:** Results show < 50% confidence  
**Solutions:**
- Move closer to microphone
- Reduce background noise
- Speak more clearly and naturally
- Ensure 3+ seconds of audio

### Recording Not Working
**Problem:** Browser cannot access microphone  
**Solutions:**
- Allow microphone permissions in browser
- Check system privacy settings
- Use HTTPS or localhost
- Try a different browser (Chrome/Firefox recommended)

---

## Quick Checklist Before Upload

- [ ] Audio is 3+ seconds long
- [ ] Only one speaker present
- [ ] No background music or noise
- [ ] File size under 32MB
- [ ] Format is WAV, MP3, OGG, M4A, FLAC, or WEBM
- [ ] Microphone is 6-12 inches from mouth
- [ ] Volume is consistent throughout

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=06b6d4,6366f1&height=120&section=footer&text=Happy+Analyzing!&fontSize=20&fontColor=fff" alt="Footer" />
</p>

<p align="center">
  <a href="https://github.com/issu321/Emotion-Detection-Using-ANN">Back to Repository</a>
</p>
