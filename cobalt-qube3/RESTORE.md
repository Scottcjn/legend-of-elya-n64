# Reviving a Sun Cobalt Qube 3 (bootable image + restore guide)

Most Cobalt Qube 3 units on the used market arrive dead: the original IDE drive
has failed and the factory OS is gone, so the machine hangs at "Loading Kernel"
and there is no way to reinstall without the long-lost Cobalt restore CD and a
BOOTP/NFS server that speaks NFSv2.

This is a ready-to-write image of a working Cobalt Linux install for the x86
Qube 3 (AMD K6 series). Write it to an SD card, put the card in an SD-to-IDE
adapter, drop it in the Qube, and it boots. It comes with SSH already working
and, if you want it, Sophia's tiny transformer (see the rest of this folder) and
the RustChain proof-of-antiquity miner.

## What you need

- A Sun Cobalt Qube 3 (x86 / AMD K6 model)
- An SD-to-IDE (44-pin 2.5" or 40-pin 3.5") adapter, or a CF-to-IDE adapter
- An SD/CF card, 4 GB or larger (a genuine card, not a fake-capacity one)
- A card reader on a modern PC

## Write the image

Download the image (see "Download" below), decompress, and write it to the card.

Linux / macOS:

    xz -d qube3-cobalt.img.xz
    sudo dd if=qube3-cobalt.img of=/dev/sdX bs=1M conv=fsync status=progress

Replace `/dev/sdX` with your card (check `lsblk` first, get this wrong and you
overwrite the wrong disk). On Windows, use balenaEtcher and point it at the
decompressed `.img`.

## First boot

1. Put the card in the adapter, install it in the Qube as the primary drive, and
   power on. The front LCD lights up and the machine boots Cobalt Linux.
2. By default it comes up on DHCP. Find the address it picked up (check your
   router, or read it off the front LCD panel).
3. SSH in. The image ships dropbear 0.53.1, which is 2011-era, so a modern SSH
   client needs legacy crypto flags:

        ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 \
            -o HostKeyAlgorithms=+ssh-rsa \
            -o Ciphers=+aes128-ctr \
            -o MACs=+hmac-sha1 \
            root@<the-qube-ip>

   Default password: `cobalt` (change it, see below).

Telnet (port 23) is also enabled as a fallback for very old clients.

## Configure it your way (drop-in text file)

The image reads `/root/qube-setup.conf` on every boot and applies whatever you
set. Edit that file (over SSH, or mount the card's root partition on a Linux box),
then reboot. A template is included as `qube-setup.conf` in this folder:

    HOSTNAME=my-qube
    NET=dhcp                 # or: NET=static 192.168.1.50 255.255.255.0 192.168.1.1
    ROOT_PASSWORD=changeme
    RTC_WALLET=your-wallet-name
    SSH_PUBKEY=ssh-ed25519 AAAA...   # optional, added to authorized_keys

Blank values keep the current setting. The applier is `firstboot.sh` in this
folder; it also regenerates unique SSH host keys on the first boot so no two
machines share a key.

## What is on the image

- Cobalt Linux (kernel 2.2.16, glibc 2.1.3, gcc 2.95) with the network, telnet,
  ftp, and web services from the factory build, plus dropbear SSH.
- The RustChain proof-of-antiquity miner (opt-in). A real K6 is a genuine retro
  x86 device and earns the 1.4x antiquity tier honestly. Set `RTC_WALLET` to your
  own wallet; leave it blank to skip mining.
- Sophia's 819K-parameter transformer (`nano_gpt.c` in this folder), the same
  engine that first ran on a Nintendo 64, generating at about 12 tokens/sec on
  the K6. Build it with `build.sh`.

## Download

The image is published on the Internet Archive:

- Details: https://archive.org/details/cobalt-qube3-bootable-image
- Direct: https://archive.org/download/cobalt-qube3-bootable-image/qube3-cobalt.img.xz

Note: the disk image is produced from a sanitized copy of a working install. The
root password is reset to the default above, the SSH host keys are removed and
regenerated on first boot, and no personal wallet or credentials are included.

## Credits and license

Built by Elyan Labs. The transformer engine is licensed as in the parent project.
The restore tooling here is AGPL-3.0. The Qube 3 and Cobalt Linux are the work of
Cobalt Networks / Sun Microsystems.
