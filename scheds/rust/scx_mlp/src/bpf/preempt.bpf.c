/* SPDX-License-Identifier: GPL-2.0 */
/*
 * To be included to the main.bpf.c
 */

/*
 * Preemption related ones
 */
struct preemption_info {
	u64		stopping_tm_est_ns;
	u64		lat_cri;
	struct cpu_ctx	*cpuc;
};

static u64 get_est_stopping_time(struct task_ctx *taskc, u64 now)
{
	return now + taskc->avg_runtime;
}

static bool can_a_preempt_b(struct preemption_info *prm_a,
			    struct preemption_info *prm_b)
{
	/*
	 * Check one's latency criticality and deadline.
	 */
	if ((prm_a->lat_cri > prm_b->lat_cri) &&
	    (prm_a->stopping_tm_est_ns < prm_b->stopping_tm_est_ns))
		return true;
	return false;
}

static bool can_task1_kick_task2(struct preemption_info *prm_task1,
				 struct preemption_info *prm_task2)
{
	/*
	 * A caller should ensure that task2 is not a lock holder.
	 */

	/*
	 * If that CPU runs a lower priority task, that's a victim
	 * candidate.
	 */
	return can_a_preempt_b(prm_task1, prm_task2);
}

static bool can_cpu1_kick_cpu2(struct preemption_info *prm_cpu1,
			       struct preemption_info *prm_cpu2,
			       struct cpu_ctx *cpuc2)
{
	/*
	 * Set a CPU information
	 */
	prm_cpu2->stopping_tm_est_ns = cpuc2->stopping_tm_est_ns;
	prm_cpu2->lat_cri = cpuc2->lat_cri;
	prm_cpu2->cpuc = cpuc2;

	/*
	 * Never preeempt a CPU running a lock holder.
	 */
	if (prm_cpu2->cpuc->lock_holder)
		return false;

	/*
	 * If that CPU runs a lower priority task, that's a victim
	 * candidate.
	 */
	return can_a_preempt_b(prm_cpu1, prm_cpu2);
}

static bool is_worth_kick_other_task(struct task_ctx *taskc)
{
	/*
	 * Preemption is not free. It is expensive involving context switching,
	 * etc. Hence, we first judiciously check whether it is worth trying to
	 * victimize another CPU as the current task is urgent enough.
	 */
	return (taskc->lat_cri >= sys_stat.thr_lat_cri);
}

static bool can_cpu_be_kicked(u64 now, struct cpu_ctx *cpuc)
{
	return cpuc->is_online;
}

static struct cpu_ctx *find_victim_cpu(const struct cpumask *cpumask,
				       struct task_ctx *taskc, u64 now)
{
	/*
	 * We see preemption as a load-balancing problem. In a system with N
	 * CPUs, ideally, the top N tasks with the highest latency priorities
	 * should run on the N CPUs all the time. This is the same as the
	 * load-balancing problem; the load-balancing problem finds a least
	 * loaded server, and the preemption problem finds a CPU running a
	 * least latency critical task. Hence, we use the 'power of two random
	 * choices' technique.
	 */
	struct cpu_ctx *cpuc;
	struct preemption_info prm_task, prm_cpus[2], *victim_cpu;
	int cpu, nr_cpus;
	int i, v = 0, cur_cpu;
	int ret;

	/*
	 * Get task's preemption information for comparison.
	 */
	prm_task.stopping_tm_est_ns = get_est_stopping_time(taskc, now);
	prm_task.lat_cri = taskc->lat_cri;
	prm_task.cpuc = cpuc = get_cpu_ctx();
	if (!cpuc) {
		scx_bpf_error("Failed to lookup the current cpu_ctx");
		goto null_out;
	}
	cur_cpu = cpuc->cpu_id;

	/*
	 * Randomly find _two_ CPUs that run lower-priority tasks than @p. To
	 * traverse CPUs in a random order, we start from a random CPU ID in a
	 * random direction (left or right). The random-order traversal helps
	 * to mitigate the thundering herd problem. Otherwise, all CPUs may end
	 * up finding the same victim CPU.
	 *
	 * In the worst case, the current logic traverses _all_ CPUs. It would
	 * be too expensive to perform every task queue. We need to revisit
	 * this if the traversal cost becomes problematic.
	 */
	barrier();
	nr_cpus = bpf_cpumask_weight(cpumask);
	bpf_for(i, 0, nr_cpus) {
		/*
		 * Decide a CPU ID to examine.
		 */
		cpu = bpf_cpumask_any_distribute(cpumask);

		if (cpu >= nr_cpu_ids || cur_cpu == cpu)
			continue;

		/*
		 * Check whether that CPU is qualified to run @p.
		 */
		cpuc = get_cpu_ctx_id(cpu);
		if (!cpuc) {
			scx_bpf_error("Failed to lookup cpu_ctx: %d", cpu);
			goto null_out;
		}

		if (!can_cpu_be_kicked(now, cpuc))
			continue;

		/*
		 * If that CPU runs a lower priority task, that's a victim
		 * candidate.
		 *
		 * Note that a task running on cpu 2 (prm_cpus[v]) cannot
		 * be a lock holder.
		 */
		ret = can_cpu1_kick_cpu2(&prm_task, &prm_cpus[v], cpuc);
		if (ret == true && ++v >= 2)
			break;
	}

	/*
	 * Choose a final victim CPU.
	 */
	switch(v) {
	case 2:	/* two candidates */
		victim_cpu = can_task1_kick_task2(&prm_cpus[0], &prm_cpus[1]) ?
				&prm_cpus[0] : &prm_cpus[1];
		goto bingo_out;
	case 1:	/* one candidate */
		victim_cpu = &prm_cpus[0];
		goto bingo_out;
	case 0:	/* no candidate */
		goto null_out;
	default:/* something wrong */
		goto null_out;
	}

bingo_out:
	return victim_cpu->cpuc;

null_out:
	return NULL;
}

static void ask_cpu_yield(struct cpu_ctx *victim_cpuc)
{
	/*
	 * Note that we avoid using scx_bpf_kick_cpu() on purpose.
	 * While scx_bpf_kick_cpu() can trigger a task preemption immediately,
	 * it incurs an expensive IPI operation. Furthermore, an IPI operation
	 * is more costly in certain processor architectures or in older
	 * generations of processors, causing performance variations among
	 * processors. Thus, let's avoid using the IPI, scx_bpf_kick_cpu(), and
	 * set the victim task's time slice to zero so the victim task yields
	 * the CPU in the next scheduling point.
	 */
	struct rq *victim_rq;
	struct task_struct *victim_p;

	victim_rq = scx_bpf_cpu_rq(victim_cpuc->cpu_id);
	if (victim_rq && (victim_p = victim_rq->curr)) {
		/*
		 * Finding a victim is racy, but we do not coordinate. Thus,
		 * two different CPUs can choose the same victim CPU. We do not
		 * coordinate this on purpose because such a race is rare, but
		 * controlling it would impose high synchronization overhead.
		 *
		 * Instead, we set the victim's stopping_tm_est_ns to zero
		 * atomically to avoid the same victim CPU can be chosen
		 * repeatedly. In addition, once updating the
		 * stopping_tm_est_ns to zero succeeds, that means this CPU
		 * wins in preempting the victim CPU. Hence, let's set the
		 * victim task's time slice to one (not zero).
		 *
		 * Why not set the time slice to zero? We avoid setting the
		 * time slice to zero because the sched_ext core in the kernel
		 * fixes the zero time slice to the default time slice
		 * (SCX_SLICE_DFL, 20 msec).
		 */
		u64 old = victim_cpuc->stopping_tm_est_ns;
		if (old) {
			bool ret = __sync_bool_compare_and_swap(
				&victim_cpuc->stopping_tm_est_ns, old, 0);
			if (ret)
				WRITE_ONCE(victim_p->scx.slice, 1);
		}
	}
}

static void try_find_and_kick_victim_cpu(struct task_struct *p,
					 struct task_ctx *taskc, u64 dsq_id)
{
	struct bpf_cpumask *cd_cpumask, *cpumask;
	struct cpdom_ctx *cpdomc;
	struct cpu_ctx *victim_cpuc;
	struct cpu_ctx *cpuc_cur;
	u64 now;

	/*
	 * Don't even try to perform expensive preemption for greedy tasks.
	 */
	if (!is_eligible(taskc))
		return;

	/*
	 * Check if it is worth to try to kick other CPU.
	 */
	if (!is_worth_kick_other_task(taskc))
		return;

	/*
	 * Prepare a cpumak so we find a victim in @p's compute domain.
	 */
	cpuc_cur = get_cpu_ctx();
	if (!cpuc_cur)
		return;

	cpumask = cpuc_cur->tmp_t_mask;
	cpdomc = MEMBER_VPTR(cpdom_ctxs, [dsq_id]);
	cd_cpumask = MEMBER_VPTR(cpdom_cpumask, [dsq_id]);
	if (!cpdomc || !cd_cpumask || !cpumask)
		return;

	bpf_cpumask_and(cpumask, cast_mask(cd_cpumask), p->cpus_ptr);

	/*
	 * Find a victim CPU among CPUs that run lower-priority tasks.
	 */
	now = scx_bpf_now();
	victim_cpuc = find_victim_cpu(cast_mask(cpumask), taskc, now);

	/*
	 * If a victim CPU is chosen, preempt the victim by kicking it.
	 */
	if (victim_cpuc) {
		ask_cpu_yield(victim_cpuc);
		cpuc_cur->nr_preempt++;
	}
}

static void reset_cpu_preemption_info(struct cpu_ctx *cpuc, bool released)
{
	if (released) {
		/*
		 * When the CPU is taken by high priority scheduler,
		 * set things impossible to preempt.
		 */
		cpuc->lat_cri = SCX_SLICE_INF;
		cpuc->stopping_tm_est_ns = 0;
	} else {
		/*
		 * When the CPU is idle,
		 * set things easy to preempt.
		 */
		cpuc->lat_cri = 0;
		cpuc->stopping_tm_est_ns = SCX_SLICE_INF;
	}
}

