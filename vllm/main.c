#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/uio.h>
#include <unistd.h>
#include <liburing.h>

int main(void)
{
    struct io_uring ring;
    struct io_uring_sqe *sqe;
    struct io_uring_cqe *cqe;

    int ret = io_uring_queue_init(8, &ring, 0);
    if (ret < 0) {
        fprintf(stderr, "io_uring_queue_init: %s (%d)\n",
                strerror(-ret), ret);
        return 1;
    }

    printf("io_uring_setup: OK\n");

    /* Exercise io_uring_register (IORING_REGISTER_PROBE) — mirrors uring_available() probe check */
    struct io_uring_probe *probe = io_uring_get_probe_ring(&ring);
    if (!probe) {
        fprintf(stderr, "io_uring_get_probe_ring: failed (IORING_REGISTER_PROBE blocked or unsupported)\n");
        return 1;
    }

    if (!io_uring_opcode_supported(probe, IORING_OP_READ)) {
        fprintf(stderr, "IORING_OP_READ: not supported\n");
        io_uring_free_probe(probe);
        return 1;
    }
    if (!io_uring_opcode_supported(probe, IORING_OP_READ_FIXED)) {
        fprintf(stderr, "IORING_OP_READ_FIXED: not supported\n");
        io_uring_free_probe(probe);
        return 1;
    }
    io_uring_free_probe(probe);

    printf("io_uring_register (PROBE): OK\n");

    /* Exercise io_uring_register (IORING_REGISTER_FILES) */
    int fd = open("/dev/null", O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    ret = io_uring_register_files(&ring, &fd, 1);
    if (ret < 0) {
        fprintf(stderr, "io_uring_register_files: %s (%d)\n",
                strerror(-ret), ret);
        return 1;
    }
    io_uring_unregister_files(&ring);

    printf("io_uring_register (FILES): OK\n");

    /* Exercise io_uring_register (IORING_REGISTER_BUFFERS) — mirrors uring_available() buffer check */
    char buf[4096];
    struct iovec iov = { buf, sizeof(buf) };
    ret = io_uring_register_buffers(&ring, &iov, 1);
    if (ret < 0) {
        fprintf(stderr, "io_uring_register_buffers: %s (%d)\n",
                strerror(-ret), ret);
        return 1;
    }
    io_uring_unregister_buffers(&ring);

    printf("io_uring_register (BUFFERS): OK\n");

    /* Exercise io_uring_enter */
    sqe = io_uring_get_sqe(&ring);
    io_uring_prep_nop(sqe);
    sqe->user_data = 0x1234;

    ret = io_uring_submit(&ring);
    if (ret < 0) {
        fprintf(stderr, "io_uring_submit: %s (%d)\n",
                strerror(-ret), ret);
        return 1;
    }

    ret = io_uring_wait_cqe(&ring, &cqe);
    if (ret < 0) {
        fprintf(stderr, "io_uring_wait_cqe: %s (%d)\n",
                strerror(-ret), ret);
        return 1;
    }

    if (cqe->res < 0) {
        fprintf(stderr, "NOP completion failed: %s\n",
                strerror(-cqe->res));
        return 1;
    }

    printf("io_uring_enter: OK\n");
    printf("io_uring NOP completion: OK\n");

    io_uring_cqe_seen(&ring, cqe);
    io_uring_queue_exit(&ring);
    close(fd);

    printf("io_uring Kubernetes test: PASS\n");
    return 0;
}
