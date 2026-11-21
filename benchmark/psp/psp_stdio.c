// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause
//
// The stdout/stderr half of pspsdk's src/debug/stdio.c, trimmed to what a
// compute binary uses. It registers a "tty" IO driver and points stdout and
// stderr at it, so newlib's printf reaches the handler installed below, which
// writes to the remote SIO port. stdin and the screen printf are dropped.

#include <pspkernel.h>
#include <pspiofilemgr.h>
#include <pspstdio.h>
#include <unistd.h>

typedef int (*NcnnPspPrintHandler)(const char* data, int len);

static int g_initialised = 0;
static NcnnPspPrintHandler g_stdout_handler = 0;
static NcnnPspPrintHandler g_stderr_handler = 0;
static SceUID g_out_sema = 0;

static int io_init(PspIoDrvArg* arg)
{
    (void)arg;
    return 0;
}

static int io_exit(PspIoDrvArg* arg)
{
    (void)arg;
    return 0;
}

static int io_open(PspIoDrvFileArg* arg, char* file, int mode, SceMode mask)
{
    (void)file;
    (void)mode;
    (void)mask;

    if ((arg->fs_num != STDOUT_FILENO) && (arg->fs_num != STDERR_FILENO))
        return (int)SCE_KERNEL_ERROR_NOFILE;

    return 0;
}

static int io_close(PspIoDrvFileArg* arg)
{
    (void)arg;
    return 0;
}

static int io_read(PspIoDrvFileArg* arg, char* data, int len)
{
    (void)arg;
    (void)data;
    (void)len;
    return 0;
}

static int io_write(PspIoDrvFileArg* arg, const char* data, int len)
{
    int ret = 0;

    (void)sceKernelWaitSema(g_out_sema, 1, 0);
    if ((arg->fs_num == STDOUT_FILENO) && (g_stdout_handler != 0))
        ret = g_stdout_handler(data, len);
    else if ((arg->fs_num == STDERR_FILENO) && (g_stderr_handler != 0))
        ret = g_stderr_handler(data, len);
    (void)sceKernelSignalSema(g_out_sema, 1);

    return ret;
}

static SceOff io_lseek(PspIoDrvFileArg* arg, SceOff ofs, int whence)
{
    (void)arg;
    (void)ofs;
    (void)whence;
    return 0;
}

static int io_ioctl(PspIoDrvFileArg* arg, unsigned int cmd, void* indata, int inlen, void* outdata, int outlen)
{
    (void)arg;
    (void)cmd;
    (void)indata;
    (void)inlen;
    (void)outdata;
    (void)outlen;
    return 0;
}

static int io_remove(PspIoDrvFileArg* arg, const char* name)
{
    (void)arg;
    (void)name;
    return 0;
}

static int io_mkdir(PspIoDrvFileArg* arg, const char* name, SceMode mode)
{
    (void)arg;
    (void)name;
    (void)mode;
    return 0;
}

static int io_rmdir(PspIoDrvFileArg* arg, const char* name)
{
    (void)arg;
    (void)name;
    return 0;
}

static int io_dopen(PspIoDrvFileArg* arg, const char* dir)
{
    (void)arg;
    (void)dir;
    return 0;
}

static int io_dclose(PspIoDrvFileArg* arg)
{
    (void)arg;
    return 0;
}

static int io_dread(PspIoDrvFileArg* arg, SceIoDirent* dir)
{
    (void)arg;
    (void)dir;
    return 0;
}

static int io_getstat(PspIoDrvFileArg* arg, const char* file, SceIoStat* stat)
{
    (void)arg;
    (void)file;
    (void)stat;
    return 0;
}

static int io_chstat(PspIoDrvFileArg* arg, const char* file, SceIoStat* stat, int bits)
{
    (void)arg;
    (void)file;
    (void)stat;
    (void)bits;
    return 0;
}

static int io_rename(PspIoDrvFileArg* arg, const char* oldname, const char* newname)
{
    (void)arg;
    (void)oldname;
    (void)newname;
    return 0;
}

static int io_chdir(PspIoDrvFileArg* arg, const char* dir)
{
    (void)arg;
    (void)dir;
    return 0;
}

static int io_mount(PspIoDrvFileArg* arg)
{
    (void)arg;
    return 0;
}

static int io_umount(PspIoDrvFileArg* arg)
{
    (void)arg;
    return 0;
}

static int io_devctl(PspIoDrvFileArg* arg, const char* devname, unsigned int cmd, void* indata, int inlen, void* outdata, int outlen)
{
    (void)arg;
    (void)devname;
    (void)cmd;
    (void)indata;
    (void)inlen;
    (void)outdata;
    (void)outlen;
    return 0;
}

static int io_unknown(PspIoDrvFileArg* arg)
{
    (void)arg;
    return 0;
}

static PspIoDrvFuncs tty_funcs = {
    io_init,
    io_exit,
    io_open,
    io_close,
    io_read,
    io_write,
    io_lseek,
    io_ioctl,
    io_remove,
    io_mkdir,
    io_rmdir,
    io_dopen,
    io_dclose,
    io_dread,
    io_getstat,
    io_chstat,
    io_rename,
    io_chdir,
    io_mount,
    io_umount,
    io_devctl,
    io_unknown,
};

static PspIoDrv tty_driver = {
    "tty", 0x10, 0x800, "TTY", &tty_funcs};

static int tty_init(void)
{
    int ret;

    (void)sceIoDelDrv("tty");
    ret = sceIoAddDrv(&tty_driver);
    if (ret < 0)
        return ret;

    g_out_sema = sceKernelCreateSema("TtyOutMutex", 0, 1, 1, 0);
    if (g_out_sema < 0)
        return g_out_sema;

    ret = sceKernelStdoutReopen("tty1:", PSP_O_WRONLY, 0777);
    if (ret < 0)
        return ret;

    ret = sceKernelStderrReopen("tty2:", PSP_O_WRONLY, 0777);
    if (ret < 0)
        return ret;

    g_initialised = 1;
    return 0;
}

int pspDebugInstallStdoutHandler(NcnnPspPrintHandler handler)
{
    if (g_initialised == 0)
    {
        int ret = tty_init();
        if (ret < 0)
            return ret;
    }

    g_stdout_handler = handler;
    return 0;
}

int pspDebugInstallStderrHandler(NcnnPspPrintHandler handler)
{
    if (g_initialised == 0)
    {
        int ret = tty_init();
        if (ret < 0)
            return ret;
    }

    g_stderr_handler = handler;
    return 0;
}
