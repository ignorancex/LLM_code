def computeDeriv(poly):
  res = []
  for i, x in enumerate(poly[1:]):
    res.append(x * (i + 1))
  print(res)
  return res or [0.0]
