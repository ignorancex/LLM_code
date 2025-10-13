#include "WolframLibrary.h"
#include "WolframIOLibraryFunctions.h"

#include <stdio.h>
#include "SSm0.h"


EXTERN_C DLLEXPORT int SSm0qq(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
  if (Argc != 3) return LIBRARY_FUNCTION_ERROR;

  mint eo = MArgument_getInteger(Args[0]);
  mreal x = MArgument_getReal(Args[1]);
  mreal y = MArgument_getReal(Args[2]);

  if (eo < -3 || x < 0 || x > 1 || y < -1 || y > 1) return LIBRARY_FUNCTION_ERROR;

  MArgument_setReal(res,SSm0_tldI(eo, x, y));

  return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int SSm0gg(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
  if (Argc != 3) return LIBRARY_FUNCTION_ERROR;

  mint eo = MArgument_getInteger(Args[0]);
  mreal x = MArgument_getReal(Args[1]);
  mreal y = MArgument_getReal(Args[2]);

  if (eo < -3 || x < 0 || x > 1 || y < -1 || y > 1) return LIBRARY_FUNCTION_ERROR;

  MArgument_setReal(res,SSm0_tldS(eo, x, y));

  return LIBRARY_NO_ERROR;
}

DLLEXPORT mint WolframLibrary_getVersion( ) {
  return WolframLibraryVersion;
}

DLLEXPORT int WolframLibrary_initialize( WolframLibraryData libData) {
  return 0;
}
