# Contributing to Legend of Elya (N64)

Thank you for your interest in contributing to Legend of Elya — an N64 game featuring a real 819K-parameter transformer running on the VR4300 MIPS III CPU!

## About This Project

Legend of Elya pushes the boundaries of what's possible on vintage hardware:
- Real transformer inference at 60 tok/s on Nintendo 64
- Zelda-style dungeon exploration
- AI-powered NPCs using byte-level neural networks
- Built with libdragon SDK

## Development Setup

### Prerequisites

- Nintendo 64 development hardware or emulator
- libdragon SDK installed
- MIPS III cross-compiler toolchain
- Python 3.8+ (for model conversion tools)

### Installing libdragon

```bash
git clone https://github.com/DragonMinded/libdragon.git
cd libdragon
make install
```

### Building the Project

1. Clone the repository:
```bash
git clone https://github.com/Scottcjn/legend-of-elya-n64.git
cd legend-of-elya-n64
```

2. Set up environment variables:
```bash
export N64_ROOT=/path/to/libdragon
export PATH=$N64_ROOT/bin:$PATH
```

3. Build the ROM:
```bash
make clean
make
```

4. The output will be `legend-of-elya.z64`

## Development Workflow

### Running on Emulator

Test your changes using an accurate N64 emulator:

```bash
# Using Ares emulator
ares legend-of-elya.z64

# Using Mupen64Plus
mupen64plus legend-of-elya.z64
```

### Testing on Real Hardware

For authentic testing on Nintendo 64:
- Use an EverDrive 64 or similar flash cart
- Transfer the built ROM to SD card
- Test on actual N64 hardware

## Code Style Guidelines

### C Code Standards

- Follow the existing code style in the project
- Use 4-space indentation
- Keep functions under 100 lines when possible
- Comment complex MIPS assembly sections

### Memory Constraints

The N64 has limited RAM (4-8MB). Be mindful of:
- Static allocations vs dynamic
- Texture memory usage
- Audio buffer sizes
- Transformer model weight storage

### Performance Considerations

- Profile on real hardware when possible
- Optimize hot paths in the inference engine
- Use fixed-point arithmetic where appropriate
- Cache-friendly data structures

## Making Contributions

### Types of Contributions Welcome

- Bug fixes for game mechanics
- Performance optimizations
- Documentation improvements
- New dungeon areas or NPCs
- Model compression techniques
- Tooling improvements

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test on emulator and/or real hardware
5. Update documentation if needed
6. Submit a pull request with clear description

### Commit Message Format

```
feat: Add new dungeon area - Crystal Caverns
fix: Correct NPC collision detection
perf: Optimize transformer inference by 15%
docs: Update build instructions for Windows
test: Add hardware compatibility tests
```

## Model Development

### Converting Models for N64

The transformer model must be converted to a format suitable for the VR4300:

```bash
python tools/convert_model.py \
    --input model.pth \
    --output assets/model.bin \
    --quantize 8bit \
    --target mips
```

### Model Constraints

- Maximum model size: ~2MB (compressed)
- Supported architectures: 819K parameters max
- Quantization: 8-bit weights recommended
- No floating-point unit on VR4300 (use fixed-point)

## Testing

### Test Checklist

Before submitting PR:
- [ ] Game boots on emulator
- [ ] No crashes during 30-minute playtest
- [ ] Transformer inference produces coherent output
- [ ] NPCs respond correctly
- [ ] Save/load functionality works
- [ ] Memory usage stays within limits

### Hardware Compatibility

Test on multiple platforms when possible:
- NTSC N64
- PAL N64
- iQue Player (China)
- Various EverDrive models

## Resources

- [libdragon Documentation](https://libdragon.dev/)
- [N64 Programming Wiki](https://n64.dev/)
- [VR4300 datasheet](https://datasheets.chipdb.org/)
- [RustChain Bounties](https://github.com/Scottcjn/rustchain-bounties) — Earn RTC for contributions

## Questions?

- Open an issue for bugs or feature requests
- Join the Discord: https://discord.gg/cafc4nDV
- Check existing issues before creating new ones

## Code of Conduct

Be respectful and constructive. This is a passion project pushing hardware limits!

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
