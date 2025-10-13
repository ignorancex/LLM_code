#include "pvn.h"

#ifdef PVN_TEST
typedef struct {
  void *addr;
  long fof;
  char *sym;
  long tot;
} pvn_addr_rec_x;

typedef struct {
  void *this_fn;
  void *call_site;
  long tns, dur;
  size_t level;
} pvn_prof_rec_x;

static int PVN_NO_PROF addr_cmp(const void *a, const void *b)
{
  PVN_ASSERT(a);
  PVN_ASSERT(b);
  if (a == b)
    return 0;
  const pvn_addr_rec_x *const A = (const pvn_addr_rec_x*)a;
  const pvn_addr_rec_x *const B = (const pvn_addr_rec_x*)b;
  if (A->addr < B->addr)
    return -1;
  if (B->addr < A->addr)
    return 1;
  return 0;
}

static ssize_t PVN_NO_PROF proct(FILE *const pt, FILE *const bt, FILE *const st, FILE *const tt)
{
  PVN_ASSERT(pt);
  PVN_ASSERT(bt);
  PVN_ASSERT(st);
  PVN_ASSERT(tt);
  pvn_addr_rec_x *addrs = (pvn_addr_rec_x*)NULL;
  PVN_SYSP_CALL(addrs = (pvn_addr_rec_x*)calloc((size_t)1u, sizeof(pvn_addr_rec_x)));
  size_t addrc = (size_t)0u;
  while (fread((addrs + addrc), (sizeof(pvn_addr_rec_t) - sizeof(char*)), (size_t)1u, bt) == (size_t)1u) {
    PVN_SYSI_CALL(fseek(st, addrs[addrc].fof, SEEK_SET));
    addrs[addrc].fof = 0l;
    addrs[addrc].sym = (char*)NULL;
    PVN_SYSI_CALL((addrs[addrc].fof = (long)getline(&(addrs[addrc].sym), (size_t*)&(addrs[addrc].fof), st)) < (ssize_t)0);
    addrs[addrc].sym[addrs[addrc].fof - 1] = '(';
    addrs[addrc].tot = addrs[addrc].fof = 0l;
    PVN_SYSP_CALL(addrs = (pvn_addr_rec_x*)realloc(addrs, (++addrc + 1u) * sizeof(pvn_addr_rec_x)));
  }
#ifndef NDEBUG
  addrs[addrc].addr = NULL;
  addrs[addrc].fof = 0l;
  addrs[addrc].sym = (char*)NULL;
  addrs[addrc].tot = 0l;
#endif /* !NDEBUG */
  qsort(addrs, addrc, sizeof(pvn_addr_rec_x), addr_cmp);
  pvn_prof_rec_x *profs = (pvn_prof_rec_x*)NULL;
  PVN_SYSP_CALL(profs = (pvn_prof_rec_x*)calloc((size_t)1u, sizeof(pvn_prof_rec_x)));
  size_t profc = (size_t)0u;
  ssize_t max_level = (ssize_t)0;
  for (size_t level = (size_t)0u; fread((profs + profc), sizeof(pvn_prof_rec_t), (size_t)1u, pt) == (size_t)1u; ) {
    pvn_addr_rec_x *addr = (addrs + addrc);
    PVN_SYSP_CALL(addr = bsearch((profs + profc), addrs, addrc, sizeof(pvn_addr_rec_x), addr_cmp));
    if (profs[profc].call_site) {
      profs[profc].call_site = (void*)(addr->sym);
      profs[profc].level = level++;
      PVN_SYSP_CALL(profs = (pvn_prof_rec_x*)realloc(profs, (++profc + 1u) * sizeof(pvn_prof_rec_x)));
    }
    else if (profc) {
      for (size_t i = (profc - 1u); ; --i) {
        if (profs[profc].this_fn == profs[i].this_fn) {
          /* assumes no wrap-around of the timer */
          profs[i].dur = (profs[profc].tns - profs[i].tns);
          ++(addr->fof);
          addr->tot += profs[i].dur;
          break;
        }
        if (!i) {
          max_level = -(ssize_t)(profc + 1u);
          goto cleanup;
        }
      }
      --level;
    }
    else {
      max_level = (ssize_t)-1;
      goto cleanup;
    }
  }
#ifndef NDEBUG
  profs[profc].this_fn = NULL;
  profs[profc].call_site = NULL;
  profs[profc].tns = 0l;
  profs[profc].dur = 0l;
  profs[profc].level = (size_t)0u;
#endif /* !NDEBUG */
  for (size_t i = 0u; i < profc; ++i) {
    if ((ssize_t)(profs[i].level) > max_level)
      max_level = (ssize_t)(profs[i].level);
    PVN_SYSI_CALL(fprintf(tt, "%20ld ns\t", profs[i].tns) < 24);
    for (size_t j = 0u; j < profs[i].level; ++j) {
      PVN_SYSI_CALL(fputc('\t', tt) == EOF);
    }
    PVN_SYSI_CALL(fprintf(tt, "%s%ld ns)\n", (const char*)(profs[i].call_site), profs[i].dur) < 8);
  }
  PVN_SYSI_CALL(fputc('\n', tt) == EOF);
 cleanup:
  free(profs);
  profs = (pvn_prof_rec_x*)NULL;
  for (size_t i = 0u; i < addrc; ++i) {
    if (max_level >= 0l) {
      PVN_SYSI_CALL(fprintf(tt, "%s%ld)\ttotal\t%ld ns\n", addrs[i].sym, addrs[i].fof, addrs[i].tot) < 16);
    }
    free(addrs[i].sym);
    addrs[i].sym = (char*)NULL;
  }
  free(addrs);
  addrs = (pvn_addr_rec_x*)NULL;
  return max_level;
}

int PVN_NO_PROF main(int argc, char *argv[])
{
#ifdef PVN_PROFILE
  (void)printf("PVN_PROFILE=%u\n", PVN_PROFILE);
#endif /* PVN_PROFILE */
  if (argc != 2) {
    (void)fprintf(stderr, "%s prefix\n", *argv);
    return EXIT_FAILURE;
  }
  char fn[20] = { '\0' };
  (void)strcpy(strncpy(fn, argv[1], 16u) + 16, ".?t");
  (void)printf("FILES=%s\n", fn);
  fn[17] = 'p';
  FILE *const pt = fopen(fn, "rb");
  PVN_SYSP_CALL(pt);
  fn[17] = 'b';
  FILE *const bt = fopen(fn, "rb");
  PVN_SYSP_CALL(bt);
  fn[17] = 's';
  FILE *const st = fopen(fn, "rb");
  PVN_SYSP_CALL(st);
  fn[17] = 't';
  FILE *const tt = fopen(fn, "w");
  PVN_SYSP_CALL(tt);
  (void)printf("DEPTH=%zd\n", proct(pt, bt, st, tt));
  PVN_SYSI_CALL(fclose(tt));
  PVN_SYSI_CALL(fclose(st));
  PVN_SYSI_CALL(fclose(bt));
  PVN_SYSI_CALL(fclose(pt));
  return EXIT_SUCCESS;
}
#else /* !PVN_TEST */
#ifdef PVN_PROFILE
#if (PVN_PROFILE > 0u)
static pthread_mutex_t prof_lock = PTHREAD_MUTEX_INITIALIZER;
#ifndef PVN_THREAD
#define PVN_THREAD __thread
#else /* PVN_THREAD */
#error PVN_THREAD already defined
#endif /* ?PVN_THREAD */
#else /* PVN_PROFILE == 0u */
#ifndef PVN_THREAD
#define PVN_THREAD
#else /* PVN_THREAD */
#error PVN_THREAD already defined
#endif /* ?PVN_THREAD */
#endif /* ?PVN_PROFILE */
static void (*on_exit_ptr)(void) = (void (*)(void))NULL;
static long tbase = 0l;
static unsigned long thr_id = 0ul;
static void *bt_root = NULL;
static FILE *bt_file = (FILE*)NULL;
static FILE *st_file = (FILE*)NULL;
static PVN_THREAD char file_name[20];
static PVN_THREAD FILE *pt_file = (FILE*)NULL;

static void PVN_NO_PROF pflerror(const char *const f, const int l)
{
  (void)fprintf(stderr, "\n%s(%d): %s\n", f, l, strerror(errno));
  (void)fflush(stderr);
}

static int PVN_NO_PROF bt_comp(const void *a, const void *b)
{
  PVN_ASSERT(a);
  PVN_ASSERT(b);
  if (a == b)
    return 0;
  const void *const *const A = (const void *const *)a;
  const void *const *const B = (const void *const *)b;
  if (*A < *B)
    return -1;
  if (*B < *A)
    return 1;
  return 0;
}

static void PVN_NO_PROF bt_action(const void *node, VISIT order, int level)
{
  if ((order == postorder) || (order == leaf)) {
    if (node) {
      pvn_addr_rec_t *const ar = *(pvn_addr_rec_t**)node;
      if (ar) {
        const size_t sl = (size_t)(ar->fof + 1);
        ar->fof = ftell(st_file);
        if (fwrite(ar->sym, sizeof(char), sl, st_file) != sl) {
          pflerror(__FILE__, __LINE__);
          return;
        }
        if (fwrite(ar, (sizeof(*ar) - sizeof(ar->sym)), (size_t)1u, bt_file) != (size_t)1u)
          pflerror(__FILE__, __LINE__);
      }
      else
        pflerror(__FILE__, -level);
    }
    else
      pflerror(__FILE__, -level);
  }
}

static void PVN_NO_PROF bt_destroy()
{
  while (bt_root) {
    pvn_addr_rec_t *const p = *(pvn_addr_rec_t**)bt_root;
    void *const r = tdelete(p, &bt_root, bt_comp);
    if (p) {
      free(p->sym);
      free(p);
    }
    if (!r) {
      pflerror(__FILE__, __LINE__);
      return;
    }
  }
}

/* It is assumed that atexit() handlers are called when exactly one thread of the process has remained running. */
static void PVN_NO_PROF on_prog_exit()
{
#if (PVN_PROFILE > 0u)
  if (pthread_mutex_lock(&prof_lock)) {
    pflerror(__FILE__, __LINE__);
    return;
  }
#endif /* PVN_PROFILE */
  file_name[17] = 's';
  if (!(st_file = fopen(file_name, "wb"))) {
    pflerror(__FILE__, __LINE__);
    goto err;
  }
  file_name[17] = 'b';
  if (!(bt_file = fopen(file_name, "wb"))) {
    pflerror(__FILE__, __LINE__);
    goto err;
  }
  twalk(bt_root, bt_action);
 err:
  if (bt_file && fclose(bt_file))
    pflerror(__FILE__, __LINE__);
  if (st_file && fclose(st_file))
    pflerror(__FILE__, __LINE__);
  if (fflush((FILE*)NULL))
    pflerror(__FILE__, __LINE__);
  bt_destroy();
#if (PVN_PROFILE > 0u)
  if (pthread_mutex_unlock(&prof_lock))
    pflerror(__FILE__, __LINE__);
  if (pthread_mutex_destroy(&prof_lock))
    pflerror(__FILE__, __LINE__);
#endif /* PVN_PROFILE */
}

static int PVN_NO_PROF bt_insert(void *addr)
{
  if (!addr) {
    pflerror(__FILE__, __LINE__);
    return -1;
  }
#if (PVN_PROFILE > 0u)
  if (pthread_mutex_lock(&prof_lock)) {
    pflerror(__FILE__, __LINE__);
    return -2;
  }
#endif /* PVN_PROFILE */
  int r = 0;
  pvn_addr_rec_t **node = (pvn_addr_rec_t**)tfind((const void*)&addr, &bt_root, bt_comp);
  if (node)
    r = 1;
  else {
    Dl_info info;
    if (!dladdr(addr, (Dl_info*)memset(&info, 0, sizeof(info)))) {
      pflerror(__FILE__, __LINE__);
      r = -3;
    }
    else {
      pvn_addr_rec_t *const ar = (pvn_addr_rec_t*)malloc(sizeof(pvn_addr_rec_t));
      if (!ar) {
        pflerror(__FILE__, __LINE__);
        r = -4;
      }
      else {
#ifndef NDEBUG
        if (addr != info.dli_saddr)
          (void)fprintf(stderr, "\n%p != %p\n", addr, info.dli_saddr);
#endif /* !NDEBUG */
        ar->addr = addr;
        ar->fof = (long)(info.dli_sname ? strlen(info.dli_sname) : (size_t)18u);
        if (!(ar->sym = (char*)malloc(ar->fof + 2))) {
          free(ar);
          pflerror(__FILE__, __LINE__);
          r = -5;
        }
        else {
          if (info.dli_sname)
            (void)strcpy(ar->sym, info.dli_sname);
          else if (sprintf(ar->sym, "0x%016lu", (uintptr_t)(info.dli_saddr ? info.dli_saddr : addr)) < 18)
            pflerror(__FILE__, __LINE__);
          (ar->sym)[ar->fof] = '\n';
          (ar->sym)[ar->fof + 1] = '\0';
          if (!(node = (pvn_addr_rec_t**)tsearch((const void*)ar, &bt_root, bt_comp))) {
            free(ar->sym);
            free(ar);
            pflerror(__FILE__, __LINE__);
            r = -6;
          }
        }
      }
    }
  }
#if (PVN_PROFILE > 0u)
  if (pthread_mutex_unlock(&prof_lock)) {
    pflerror(__FILE__, __LINE__);
    r = -7;
  }
#endif /* PVN_PROFILE */
  return r;
}

PVN_EXTERN_C void PVN_NO_PROF __cyg_profile_func_enter(void *const this_fn, void *const call_site)
{
  bool once = false;
  if (!pt_file) {
#if (PVN_PROFILE > 0u)
    if (pthread_mutex_lock(&prof_lock)) {
      pflerror(__FILE__, __LINE__);
      return;
    }
#endif /* PVN_PROFILE */
    if (sprintf(file_name, "%016lx.pt", thr_id) < 19)
      pflerror(__FILE__, __LINE__);
    else {
      if (!(pt_file = fopen(file_name, "wb")))
        pflerror(__FILE__, __LINE__);
      else {
        ++thr_id;
#ifndef NDEBUG
        if (thr_id >
#if (PVN_PROFILE > 0u)
            PVN_PROFILE
#else /* PVN_PROFILE == 0u */
            1u
#endif /* ?PVN_PROFILE */
            )
          (void)fprintf(stderr, "\nmore threads created than anticipated\n");
#endif /* !NDEBUG */
        if (!on_exit_ptr) {
          if (atexit(on_prog_exit))
            pflerror(__FILE__, __LINE__);
          else {
            on_exit_ptr = on_prog_exit;
            struct timespec tv = { (time_t)0, 0l };
            if (clock_gettime(PVN_CLOCK_MONOTONIC, &tv))
              pflerror(__FILE__, __LINE__);
            tbase = (tv.tv_sec * 1000000000l + tv.tv_nsec);
            once = true;
          }
        }
      }
    }
#if (PVN_PROFILE > 0u)
    if (pthread_mutex_unlock(&prof_lock))
      pflerror(__FILE__, __LINE__);
#endif /* PVN_PROFILE */
  }

  if (bt_insert(this_fn) < 0)
    return;

  if (pt_file) {
    pvn_prof_rec_t pr = { this_fn, call_site, 0l };
    if (!once) {
      struct timespec tv = { (time_t)0, 0l };
      if (clock_gettime(PVN_CLOCK_MONOTONIC, &tv))
        pflerror(__FILE__, __LINE__);
      pr.tns = ((tv.tv_sec * 1000000000l + tv.tv_nsec) - tbase);
    }
    if (fwrite(&pr, sizeof(pr), 1, pt_file) != 1)
      pflerror(__FILE__, __LINE__);
  }
}

PVN_EXTERN_C void PVN_NO_PROF __cyg_profile_func_exit(void *const this_fn, void *const call_site)
{
  if (pt_file) {
    pvn_prof_rec_t pr = { this_fn, NULL /* a marker for func_exit */, (long)call_site /* just to use the argument */};
    struct timespec tv = { (time_t)0, 0l };
    if (clock_gettime(PVN_CLOCK_MONOTONIC, &tv))
      pflerror(__FILE__, __LINE__);
    pr.tns = ((tv.tv_sec * 1000000000l + tv.tv_nsec) - tbase);
    if (fwrite(&pr, sizeof(pr), 1, pt_file) != 1)
      pflerror(__FILE__, __LINE__);
  }
}
#endif /* PVN_PROFILE */
#endif /* ?PVN_TEST */
