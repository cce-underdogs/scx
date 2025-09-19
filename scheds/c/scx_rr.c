/* SPDX-License-Identifier: GPL-2.0 */

#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include <libgen.h>
#include <bpf/bpf.h>
#include <scx/common.h>
#include "scx_rr.bpf.skel.h"

const char help_fmt[] =
"Empty sched_ext scheduler templatey.\n"
"\n"
"See the top-level comment in .bpf.c for more details.\n"
"\n"
"Usage: %s [-f] [-v]\n"
"\n"
"  -p            Enable per-CPU DSQs\n"
"  -w            Prioritize waker's CPU on wakeup\n"
"  -v            Print libbpf debug messages\n"
"  -h            Display this help and exit\n";

static bool verbose;
static volatile int exit_req;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	if (level == LIBBPF_DEBUG && !verbose)
		return 0;
	return vfprintf(stderr, format, args);
}

static void sigint_handler(int rr)
{
	exit_req = 1;
}

int main(int argc, char **argv)
{
	struct scx_rr *skel;
	struct bpf_link *link;
	__u32 opt;
	__u64 ecode;

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sigint_handler);
	signal(SIGTERM, sigint_handler);
restart:
	skel = SCX_OPS_OPEN(rr_ops, scx_rr);

	while ((opt = getopt(argc, argv, "pwvh")) != -1) {
		switch (opt) {
		case 'p':
			skel->rodata->pcpu_dsq = true;
			break;
		case 'w':
			skel->rodata->use_waker = true;
			break;
		case 'v':
			verbose = true;
			break;
		default:
			fprintf(stderr, help_fmt, basename(argv[0]));
			return opt != 'h';
		}
	}

	SCX_OPS_LOAD(skel, rr_ops, scx_rr, uei);
	link = SCX_OPS_ATTACH(skel, rr_ops, scx_rr);

	fprintf(stderr, "rr scheduler is running\n");
	while (!exit_req && !UEI_EXITED(skel, uei))
		sleep(1);

	bpf_link__destroy(link);
	ecode = UEI_REPORT(skel, uei);
	scx_rr__destroy(skel);

	if (UEI_ECODE_RESTART(ecode))
		goto restart;
	return 0;
}
