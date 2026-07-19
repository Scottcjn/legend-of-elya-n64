#!/bin/sh
# Cobalt Qube 3 first-boot / every-boot configurator.
#
# Reads /root/qube-setup.conf and applies it. It is idempotent: safe to run on
# every boot, so a user just edits the config file and reboots. In the
# distributable image this is invoked early from /etc/rc.d/rc.sysinit.
#
# Written for the stock Cobalt userland (kernel 2.2, glibc 2.1.3, old net-tools,
# no /etc/shadow, no chpasswd). Passwords are MD5 ($1$) hashes in /etc/passwd.

CONF=/root/qube-setup.conf
LOG=/var/log/qube-firstboot.log
DBDIR=/etc/dropbear

log() { echo "`date` $*" >> "$LOG" 2>/dev/null; }

# --- 1. Unique SSH host keys (removed from the sanitized image) --------------
if [ ! -f "$DBDIR/dropbear_rsa_host_key" ] && [ -x /usr/bin/dropbearkey ]; then
    mkdir -p "$DBDIR"
    /usr/bin/dropbearkey -t rsa -s 1024 -f "$DBDIR/dropbear_rsa_host_key" >> "$LOG" 2>&1
    /usr/bin/dropbearkey -t dss -f "$DBDIR/dropbear_dss_host_key" >> "$LOG" 2>&1
    log "generated fresh dropbear host keys"
fi

# --- 2. dropbear needs root's shell listed in /etc/shells -------------------
[ -f /etc/shells ] || printf '/bin/sh\n/bin/bash\n' > /etc/shells

# --- 3. Apply the user config, if any ---------------------------------------
[ -f "$CONF" ] || { log "no $CONF; keeping defaults"; exit 0; }

getval() { grep -i "^$1=" "$CONF" 2>/dev/null | head -1 | cut -d= -f2- | sed 's/^ *//; s/ *$//; s/\r//'; }

HOSTNAME=`getval HOSTNAME`
NET=`getval NET`
ROOT_PASSWORD=`getval ROOT_PASSWORD`
RTC_WALLET=`getval RTC_WALLET`
SSH_PUBKEY=`getval SSH_PUBKEY`

# hostname (live + persisted the Cobalt/RedHat way)
if [ -n "$HOSTNAME" ]; then
    hostname "$HOSTNAME" 2>/dev/null
    if [ -f /etc/sysconfig/network ]; then
        grep -v '^HOSTNAME=' /etc/sysconfig/network > /tmp/net.$$ 2>/dev/null
        echo "HOSTNAME=$HOSTNAME" >> /tmp/net.$$
        mv /tmp/net.$$ /etc/sysconfig/network
    fi
    log "hostname=$HOSTNAME"
fi

# network: "dhcp" (leave the normal boot path alone) or "static ip mask gw"
if [ -n "$NET" ]; then
    set -- $NET
    if [ "$1" = "static" ] && [ -n "$2" ]; then
        IP=$2; MASK=${3:-255.255.255.0}; GW=$4
        ifconfig eth0 "$IP" netmask "$MASK" up 2>/dev/null
        [ -n "$GW" ] && route add default gw "$GW" 2>/dev/null
        log "net=static $IP/$MASK gw=$GW"
    else
        log "net=dhcp (handled by normal boot)"
    fi
fi

# root password -> MD5 ($1$) hash in field 2 of /etc/passwd
if [ -n "$ROOT_PASSWORD" ]; then
    SALT=`head -c 16 /dev/urandom 2>/dev/null | md5sum 2>/dev/null | cut -c1-8`
    [ -z "$SALT" ] && SALT=`date +%s 2>/dev/null | tail -c 9`
    HASH=`perl -e 'print crypt($ARGV[0], "\$1\$".$ARGV[1]."\$")' "$ROOT_PASSWORD" "$SALT" 2>/dev/null`
    if [ -n "$HASH" ]; then
        awk -F: -v h="$HASH" 'BEGIN{OFS=":"} $1=="root"{$2=h} {print}' /etc/passwd > /etc/passwd.new 2>/dev/null \
            && mv /etc/passwd.new /etc/passwd && log "root password updated"
    else
        log "WARN: could not hash root password (perl/crypt unavailable)"
    fi
fi

# SSH public key
if [ -n "$SSH_PUBKEY" ]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    grep -qF "$SSH_PUBKEY" /root/.ssh/authorized_keys 2>/dev/null || echo "$SSH_PUBKEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    log "ssh pubkey installed"
fi

# RustChain wallet: written where the miner start line reads it
if [ -n "$RTC_WALLET" ]; then
    echo "$RTC_WALLET" > /root/.rtc_wallet
    log "rtc wallet set: $RTC_WALLET"
fi

exit 0
