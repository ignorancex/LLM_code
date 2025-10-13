#include "pvn.h"

#ifdef PVN_TEST
static int test_rolcyc(const int id, const int n)
{
  pvn_cjs_rolcyc js;
  int ret = pvn_cjs_init(&js, id, n);
  if (ret <= 0)
    return ret;
  int arr[2];
  (void)fprintf(stdout, "{\n\t");
  for (int stp = 0; stp < ret; ++stp) {
    const int r = pvn_cjs_next(&js, (int*)arr);
    if (r != 1) {
      ret = -r;
      goto end;
    }
    (void)fprintf(stdout, "{%3d,%3d},", arr[0], arr[1]);
  }
  (void)fprintf(stdout, "\b \n}\n");
 end:
  (void)pvn_cjs_free(&js);
  return ret;
}

static int test_maneb2(const int id, const int n)
{
  pvn_cjs_maneb2 js;
  int ret = pvn_cjs_init(&js, id, n);
  if (ret <= 0)
    return ret;
  (void)fprintf(stdout, "{\n\t");
  const int n_2 = n >> 1;
  if (id & 1) { /* comm */
    int arr[n_2][2][2];
    for (int stp = 0; stp < ret; ++stp) {
      const int r = pvn_cjs_next(&js, (int*)arr);
      if (r != n_2) {
        ret = -r;
        goto end;
      }
      (void)fprintf(stdout, "{\n\t\t");
      for (int p = 0; p < r; ++p)
        (void)fprintf(stdout, "{%3d,%3d}:[%3d,%3d],", arr[p][0][0], arr[p][0][1], arr[p][1][0], arr[p][1][1]);
      (void)fprintf(stdout, "\b \n\t},");
    }
    (void)fprintf(stdout, "\b \n}\n");
  }
  else { /* no comm */
    int arr[n_2][2];
    for (int stp = 0; stp < ret; ++stp) {
      const int r = pvn_cjs_next(&js, (int*)arr);
      if (r != n_2) {
        ret = -r;
        goto end;
      }
      (void)fprintf(stdout, "{\n\t\t");
      for (int p = 0; p < r; ++p)
        (void)fprintf(stdout, "{%3d,%3d},", arr[p][0], arr[p][1]);
      (void)fprintf(stdout, "\b \n\t},");
    }
    (void)fprintf(stdout, "\b \n}\n");
  }
 end:
  (void)pvn_cjs_free(&js);
  return ret;
}

static int test_modmod(const int id, const int n)
{
  pvn_cjs_modmod js;
  int ret = pvn_cjs_init(&js, id, n);
  if (ret <= 0)
    return ret;
  (void)fprintf(stdout, "{\n\t");
  const int n_2 = n >> 1;
  if (id & 1) { /* comm */
    int arr[n_2][2][2][2];
    for (int stp = 0; stp < ret; ++stp) {
      const int r = pvn_cjs_next(&js, (int*)arr);
      if (r != n_2) {
        ret = -r;
        goto end;
      }
      (void)fprintf(stdout, "{\n\t\t");
      for (int p = 0; p < r; ++p)
        (void)fprintf(stdout, "{%3d,%3d}:[%3d,%3d],", arr[p][0][0][0], arr[p][0][1][0], arr[p][1][0][0], arr[p][1][1][0]);
      (void)fprintf(stdout, "\b \n\t},");
    }
    (void)fprintf(stdout, "\b \n}\n");
  }
  else { /* no comm */
    int arr[n_2][2][2];
    for (int stp = 0; stp < ret; ++stp) {
      const int r = pvn_cjs_next(&js, (int*)arr);
      if (r != n_2) {
        ret = -r;
        goto end;
      }
      (void)fprintf(stdout, "{\n\t\t");
      for (int p = 0; p < r; ++p)
        (void)fprintf(stdout, "{%3d,%3d},", arr[p][0][0], arr[p][1][0]);
      (void)fprintf(stdout, "\b \n\t},");
    }
    (void)fprintf(stdout, "\b \n}\n");
  }
 end:
  (void)pvn_cjs_free(&js);
  return ret;
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    (void)fprintf(stderr, "%s id n\n", argv[0]);
    return EXIT_FAILURE;
  }
  const int id = atoi(argv[1]);
  const int n = atoi(argv[2]);
  if (n <= 1)
    return EXIT_FAILURE;
  int ret = 0;
  switch (id & ~1) {
  case 0:
    ret = test_rolcyc(id, n);
    break;
  case 2:
    ret = test_maneb2(id, n);
    break;
  case 4:
    ret = test_modmod(id, n);
    break;
  default:
    return EXIT_FAILURE;
  }
  (void)fprintf(((ret <= 0) ? stderr : stdout), "ret = %d\n", ret);
  return ((ret <= 0) ? EXIT_FAILURE : EXIT_SUCCESS);
}
#else /* !PVN_TEST */
#include "../inc/pvn_cjs_ME.h"

static int *me_dup(const int n1, const int *const tbl)
{
  if (!tbl)
    return (int*)NULL;
  if (n1 < 2)
    return (int*)NULL;
  if (n1 & 1)
    return (int*)NULL;

  const int s1 = (n1 - 1);
  const int p1 = (n1 >> 1);
  const int n2 = (n1 << 1);
  /* const int s2 = (n2 - 1); */
  const int p2 = (n2 >> 1);

  int (*const tbl2)[2] = (int (*)[2])malloc((n2 - 1) * (n2 * sizeof(int)));
  if (!tbl2)
    return (int*)NULL;
  const int (*const tbl1)[2] = (const int (*)[2])tbl;

  /* (NE,SW); (NW,SE) */
  for (int p = 0; p < s1; ++p) {
    const int q = (p << 1);
    for (int a = 0; a < p1; ++a) {
      const int b = (a << 1);
      const int k = (p * p1 + a);
      const int l = (q * p2 + b);
      const int m = ((q + 1) * p2 + b);
      const int i = tbl1[k][0];
      const int j = tbl1[k][1];
      const int i2 = (i << 1);
      const int j2 = (j << 1);
      /* NE */
      tbl2[l][0] = i2;
      tbl2[l][1] = (j2 + 1);
      /* SW */
      tbl2[l + 1][0] = (i2 + 1);
      tbl2[l + 1][1] = j2;
      /* NW */
      tbl2[m][0] = i2;
      tbl2[m][1] = j2;
      /* SE */
      tbl2[m + 1][0] = (i2 + 1);
      tbl2[m + 1][1] = (j2 + 1);
    }
  }
  /* super-diag */
  const int d = ((n2 - 2) * p2);
  for (int i = 0; i < p2; ++i)
    tbl2[d + i][1] = ((tbl2[d + i][0] = (i << 1)) + 1);

  return &(tbl2[0][0]);
}

static int me_base(const int n, int *const p2)
{
  if (!p2)
    return -2;
  *p2 = 0;

  if (n < 2)
    return -1;
  if (n & 1)
    return -1;

  for (int i = (n >> 1); i; i >>= 1) {
    if (i & 1)
      return (i << 1);
    ++*p2;
  }

  return 0;
}

static int *me_p2(const int n)
{
  int p2 = 0;
  const int b = me_base(n, &p2);
  if (b <= 0)
    return (int*)NULL;

  const int *bp = (const int*)NULL;
  if (b == 2)
    bp = &(ME02[0][0][0]);
  else if (b == 6)
    bp = &(ME06[0][0][0]);
  else if (b == 10)
    bp = &(ME10[0][0][0]);
  else if (b == 14)
    bp = &(ME14[0][0][0]);
  else if (b == 18)
    bp = &(ME18[0][0][0]);
  else if (b == 22)
    bp = &(ME22[0][0][0]);
  else if (b == 26)
    bp = &(ME26[0][0][0]);
  else if (b == 30)
    bp = &(ME30[0][0][0]);
  else if (b == 34)
    bp = &(ME34[0][0][0]);
  else if (b == 38)
    bp = &(ME38[0][0][0]);
  else if (b == 42)
    bp = &(ME42[0][0][0]);
  else /* no such base table */
    return (int*)NULL;

  const size_t bsz = (bp ? ((b - 1) * (b * sizeof(int))) : (size_t)0u);
  int *cur = (int*)(bsz ? malloc(bsz) : NULL);
  if (!cur)
    return cur;
  (void)memcpy(cur, bp, bsz);
  int *prev = (int*)NULL;

  for (int i = 0; i < p2; ++i) {
    prev = cur;
    cur = me_dup((b << i), cur);
    free(prev);
    if (!cur)
      break;
  }

  return cur;
}

int pvn_cjs_init(void *const js, const int id, const int n)
{
  if (!js)
    return -1;
  if (id < 0)
    return -2;
  if (n < 2)
    return -3;
  int info = 0;

  if (!(id & ~1)) { /* row/col-cyclic */
    (void)memset(js, 0, sizeof(pvn_cjs_rolcyc));
    if (n & 1) /* n odd */
      info = n * ((n - 1) >> 1);
    else /* n even */
      info = (n >> 1) * (n - 1);
  }
  else if ((id & ~1) == 2) { /* Mantharam-Eberlein */
    pvn_cjs_maneb2 *const me2 = (pvn_cjs_maneb2*)memset(js, 0, sizeof(pvn_cjs_maneb2));
    const int *const cur = me_p2(n);
    if (!cur)
      return -3;
    me2->nxt = me2->tbl = cur;
    info = n - 1;
  }
  else if ((id & ~1) == 4) { /* modified modulus */
    (void)memset(js, 0, sizeof(pvn_cjs_modmod));
    if (n & 1) /* n odd */
      return -3;
    info = n;
  }
  else
    return -2;

  ((int*)js)[0] = id;
  ((int*)js)[1] = n;
  return info;
}

int PVN_FABI(pvn_cjs_init,PVN_CJS_INIT)(void **const ret, const int *const id, const int *const n)
{
  PVN_ASSERT(ret);
  PVN_ASSERT(id);
  PVN_ASSERT(n);
  *ret = NULL;
  switch (*id) {
  case 0:
  case 1:
    *ret = malloc(sizeof(pvn_cjs_rolcyc));
    break;
  case 2:
  case 3:
    *ret = malloc(sizeof(pvn_cjs_maneb2));
    break;
  case 4:
  case 5:
    *ret = malloc(sizeof(pvn_cjs_modmod));
    break;
  default:
    return -2;
  }
  if (!*ret)
    return -1;
  const int i = pvn_cjs_init(*ret, *id, *n);
  if (i < 0) {
    free(*ret);
    *ret = NULL;
  }
  return i;
}

int pvn_cjs_next(void *const js, int *const arr)
{
  if (!js)
    return -1;
  if (!arr)
    return -2;
  int info = 0;

  if (!*(const int*)js) { /* row-cyclic */
    pvn_cjs_rolcyc *const row = (pvn_cjs_rolcyc*)js;
    int (*const pairs)[2] = (int (*)[2])arr;

    if (++(row->q) >= (row->n)) {
      if (++(row->p) >= (row->n - 1)) {
        row->p = 0;
        row->q = 1;
      }
      else
        row->q = row->p + 1;
    }
    pairs[0][0] = row->p;
    pairs[0][1] = row->q;

    info = 1;
  }
  else if (*(const int*)js == 1) { /* column-cyclic */
    pvn_cjs_rolcyc *const col = (pvn_cjs_rolcyc*)js;
    int (*const pairs)[2] = (int (*)[2])arr;

    if (++(col->p) >= (col->q)) {
      if (++(col->q) >= (col->n)) {
        col->p = 0;
        col->q = 1;
      }
      else
        col->p = 0;
    }

    pairs[0][0] = col->p;
    pairs[0][1] = col->q;

    info = 1;
  }
  else if (*(const int*)js == 2) { /* Mantharam-Eberlein, no comm */
    /* [RANK][p/q] */
    pvn_cjs_maneb2 *const me2 = (pvn_cjs_maneb2*)js;
    int (*const pairs)[2] = (int (*)[2])arr;

    const int n_2 = me2->n >> 1;
    for (int i = 0; i < n_2; ++i) {
      for (int k = 0; k < 2; ++k) {
        pairs[i][k] = *(me2->nxt);
        ++(me2->nxt);
      }
    }

    if (++(me2->stp) >= (me2->n - 1)) {
      me2->stp = 0;
      ++(me2->swp);
      me2->nxt = me2->tbl;
    }

    info = n_2;
  }
  else if (*(const int*)js == 3) { /* Mantharam-Eberlein */
    /* [RANK][0=PAIR,1=COMM][p/q] */
    pvn_cjs_maneb2 *const me2 = (pvn_cjs_maneb2*)js;
    int (*const pairs)[2][2] = (int (*)[2][2])arr;

    const int *const cur = me2->nxt;
    const int n_2 = me2->n >> 1;
    for (int i = 0; i < n_2; ++i) {
      for (int k = 0; k < 2; ++k) {
        pairs[i][0][k] = *(me2->nxt);
        ++(me2->nxt);
      }
    }

    if (++(me2->stp) >= (me2->n - 1)) {
      me2->stp = 0;
      ++(me2->swp);
      me2->nxt = me2->tbl;
    }

    /* communication */
    for (int i = 0; i < me2->n; ++i) {
      for (int j = 0; j < me2->n; ++j) {
        /* cur[i] sent to nxt[j] */
        if (cur[i] == (me2->nxt)[j]) {
          const int k_1 = i & 1;
          const int k_2 = i >> 1;
          const int l_1 = j & 1;
          const int l_2 = j >> 1;
          const int snd = l_2 + 1;
          pairs[k_2][1][k_1] = (l_1 ? snd : -snd);
        }
      }
    }

    info = n_2;
  }
  else if (*(const int*)js == 4) { /* modified modulus, no comm */
    /* [RANK][p/q][0=PAIR,1=SHADOW] */
    pvn_cjs_modmod *const mom = (pvn_cjs_modmod*)js;
    int (*const ij)[2][2] = (int (*)[2][2])arr;

    const int n_2 = mom->n >> 1;
    const int n_1 = mom->n - 1;

    if (!(mom->stp) && !(mom->swp)) { /* init */
      for (int r = 0; r < n_2; ++r) {
        ij[r][0][1] = ij[r][0][0] = r;
        ij[r][1][1] = ij[r][1][0] = (n_1 - r);
      }
    }
    else { /* step */
      for (int r = 0; r < n_2; ++r) {
        if ((ij[r][0][1] + ij[r][1][1]) >= n_1) {
          if (++(ij[r][0][1]) == ij[r][1][1])
            ij[r][1][1] = (ij[r][0][1] -= n_2);
          ij[r][0][0] = ij[r][0][1];
        }
        else
          ij[r][1][0] = ++(ij[r][1][1]);
      }
    }

    if (++(mom->stp) >= mom->n) {
      mom->stp = 0;
      ++(mom->swp);
    }
    info = n_2;
  }
  else if (*(const int*)js == 5) { /* modified modulus */
    /* [RANK][0=PAIR,1=COMM][p/q][0=PAIR,1=SHADOW] */
    pvn_cjs_modmod *const mom = (pvn_cjs_modmod*)js;
    int (*const ij)[2][2][2] = (int (*)[2][2][2])arr;

    const int n_2 = mom->n >> 1;
    const int n_1 = mom->n - 1;

    if (!(mom->stp) && !(mom->swp)) { /* init */
      for (int r = 0; r < n_2; ++r) {
        ij[r][0][0][1] = ij[r][0][0][0] = r;
        ij[r][0][1][1] = ij[r][0][1][0] = (n_1 - r);
      }
    }
    else { /* step */
      for (int r = 0; r < n_2; ++r) {
        if ((ij[r][0][0][1] + ij[r][0][1][1]) >= n_1) {
          if (++(ij[r][0][0][1]) == ij[r][0][1][1])
            ij[r][0][1][1] = (ij[r][0][0][1] -= n_2);
          ij[r][0][0][0] = ij[r][0][0][1];
        }
        else
          ij[r][0][1][0] = ++(ij[r][0][1][1]);
      }
    }

    for (int r = 0; r < n_2; ++r) {
      ij[r][1][0][0] = ij[r][0][0][0];
      ij[r][1][0][1] = ij[r][0][0][1];
      ij[r][1][1][0] = ij[r][0][1][0];
      ij[r][1][1][1] = ij[r][0][1][1];
    }

    for (int r = 0; r < n_2; ++r) {
      if ((ij[r][1][0][1] + ij[r][1][1][1]) >= n_1) {
        if (++(ij[r][1][0][1]) == ij[r][1][1][1])
          ij[r][1][1][1] = (ij[r][1][0][1] -= n_2);
        ij[r][1][0][0] = ij[r][1][0][1];
      }
      else
        ij[r][1][1][0] = ++(ij[r][1][1][1]);
    }

    for (int r = 0; r < n_2; ++r) {
      ij[r][1][0][1] = ij[r][1][0][0];
      ij[r][1][0][0] = 0;
      ij[r][1][1][1] = ij[r][1][1][0];
      ij[r][1][1][0] = 0;
    }

    if (++(mom->stp) >= mom->n) {
      mom->stp = 0;
      ++(mom->swp);
    }
    info = n_2;

    /* communication */
    for (int i = 0; i < mom->n; ++i) {
      for (int j = 0; j < mom->n; ++j) {
        const int Is1 = i >> 1;
        const int Ia1 = i & 1;
        const int Js1 = j >> 1;
        const int Ja1 = j & 1;
        const int cur = ij[Is1][0][Ia1][0];
        const int nxt = ij[Js1][1][Ja1][1];
        /* cur sent to nxt */
        if (cur == nxt) {
          const int k_1 = Ia1;
          const int k_2 = Is1;
          const int l_1 = Ja1;
          const int l_2 = Js1;
          const int snd = l_2 + 1;
          ij[k_2][1][k_1][0] = (l_1 ? snd : -snd);
        }
      }
    }

    for (int r = 0; r < n_2; ++r)
      ij[r][1][1][1] = ij[r][1][0][1] = 0;
  }
  else
    info = 0;

  return info;
}

int PVN_FABI(pvn_cjs_next,PVN_CJS_NEXT)(void *const *const js, int *const arr)
{
  PVN_ASSERT(js);
  PVN_ASSERT(arr);
  return pvn_cjs_next(*js, arr);
}

int pvn_cjs_free(void *const js)
{
  if (!js)
    return -1;
  if ((*(const int*)js & ~1) == 2)
    free((void*)(((pvn_cjs_maneb2*)js)->tbl));
  return 0;
}

int PVN_FABI(pvn_cjs_free,PVN_CJS_FREE)(void **const js)
{
  PVN_ASSERT(js);
  const int i = pvn_cjs_free(*js);
  if (!i)
    free(*js);
  *js = NULL;
  return i;
}
#endif /* ?PVN_TEST */
