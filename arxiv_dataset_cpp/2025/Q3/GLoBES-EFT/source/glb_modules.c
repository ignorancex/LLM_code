/* GLoBES -- General LOng Baseline Experiment Simulator
 * (C) 2002 - 2007,  The GLoBES Team
 *
 * GLoBES is mainly intended for academic purposes. Proper
 * credit must be given if you use GLoBES or parts of it. Please
 * read the section 'Credit' in the README file.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
#if HAVE_CONFIG_H   /* config.h should come before any other includes */
#  include "config.h"
#endif

#include <ltdl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <globes/globes.h>
#include <globes/glb-modules.h>
#include "glb_path.h"
#include "glb_error.h"

/* Init function to be called by glbInit */
void glb_init_module_support()
{
  int s;
  s=lt_dlinit();
  if(s!=0) {glb_fatal("Could not initialize module support");exit(1);}
  return;
}

/* Finalizer for each modules */
static int finalize_module(lt_dlhandle handle, lt_ptr data)
{
  void (*clean)(void);
  clean=lt_dlsym(handle,"glb_module_clean");
  if(clean) clean();
  return 0;
}


/* Cleanup function to be called by final_clean */
void glb_close_module_support()
{
  /* This ensures that if a clean-up function is present it will be called
   * by final_clean on exit
   */
  lt_dlforeach(finalize_module,NULL);
  lt_dlexit();
}

/* Set up path to be searched for modules */
int glb_setup_module_search_path()
{

  int i,s=1;
  char *in_path;
  char *path;
  char **path_vector;
  size_t path_vector_length=0;

  /* Get the path from the environment */
  in_path=getenv("GLB_MODULES");
  if(in_path==NULL) path=NULL;
  else path=strdup(in_path);

  /* Parse the path -- try this aloud twenty times as fast as you can ;-) */
  s=glb_break_up_path(path,&path_vector,&path_vector_length);
  glb_free(path);

  /* Setup the correct directory to find the builtin modules */
  s += lt_dladdsearchdir(GLB_MODULE_DIR);

  /* Adding the elements of path vector to the module search path
   * ommiting the first (which would be the local directory) */
  for(i=1;i<path_vector_length;i++) s += lt_dladdsearchdir(path_vector[i]);

  /* cleaning up */
  for(i=0;i<path_vector_length;i++)
    glb_free((char *) path_vector[i]);
  glb_free((char **) path_vector);

  return s;
}

#define GLB_MODULE_NOT_FOUND -2
#define GLB_MODULE_NO_INIT -1

/* Checking wether a module exists and is a GLoBES module
 * The idea is to adjust verbosity level to whatever the
 * user needs, if he wants to avoid later error messages
 * he has to explicitely call glb_probe_module before any
 * other action on the module is performed.
 */
int glbProbeModule(const char *module_name, int verbosity)
{
  int *id;
  int (*init)(void);
  void (*clean)(void);
  const char *error;
  lt_dlhandle module;
  int s=0;
  /* opening the module */
  module = lt_dlopenext(module_name);
  error=lt_dlerror();
  if(!module) {
  if(verbosity>1){
      fprintf(stderr,"glb_probe_module: ERROR:"
	      " Couldn not load module %s: %s\n",
	      module_name,error);
    }
  return GLB_MODULE_NOT_FOUND;}
  if(verbosity>2) {
    fprintf(stderr,"glb_probe_module: message: module %s loaded...\n",
	    module_name);
  }

  /* getting the init symbol */
  init=lt_dlsym(module, "glb_module_init");
  error=lt_dlerror();
  if(error) {
    if(verbosity>1){
      fprintf(stderr,"glb_probe_module: ERROR:"
	      " Could not find symbol `glb_module_init'\n");
      fprintf(stderr,"glb_probe_module: ERROR: in module %s: %s\n",
	      module_name,error);
      fprintf(stderr,"glb_probe_module: ERROR: %s is not a GLoBES module\n",
	      module_name);
      lt_dlclose(module);
      if(verbosity>2) {
	fprintf(stderr,"glb_probe_module: message: unloaded module %s.\n"
		,module_name);
      }

    }
  return GLB_MODULE_NO_INIT;}
  if(verbosity>2) {
    fprintf(stderr,"glb_probe_module: message: glb_module_init found ...\n");
  }
  s=init();
  if(verbosity>2) {
    fprintf(stderr,"glb_probe_module: message: glb_module_init returns "
	    "%d ...\n",s);
  }

  /* Checking the module id */
  id=lt_dlsym(module,"glb_module_id");
  if(id) {
    s=*id;
    if(verbosity>2) {
      fprintf(stderr,"glb_probe_module: message: glb_module_id is "
	      "%d ...\n",s);
    }
  }
  else
    {
      if(verbosity>2) {
	fprintf(stderr,"glb_probe_module: ERROR no glb_module_id defined in"
		"module %s.\n" ,module_name);
      }
      s=-1;
    }


  /* Closing the module */
  clean=lt_dlsym(module,"glb_module_clean");
  if(clean!=NULL) clean();
  lt_dlclose(module);

  if(verbosity>2) {
    fprintf(stderr,"glb_probe_module: message: unloaded module %s.\n"
	    ,module_name);
  }

  return s;
}

/* Canonical open/close functions */

glb_dlhandle glbOpenModule(const char *module_name)
{
  const char *error;
  int (*init)();
  lt_dlhandle module;
  module = lt_dlopenext(module_name);
  error=lt_dlerror();
  if(!module) {
    glb_error("Could not open module!");
    fprintf(stderr,"\t while opening %s\n",module_name);
    fprintf(stderr,"\tlt_dlopen reports following error %s\n",error);
    return NULL;
  }
  init=lt_dlsym(module, "glb_module_init");
// init=lt_dlsym(module, "glb_prior_module_LTX_glb_module_init");
 if(init==NULL) {
   glb_error("Deficient module");
   fprintf(stderr,"\tmodule %s has no working glb_module_init!\n",module_name);
   return NULL;
 }

 init();

 return (glb_dlhandle) module;
}

int glbCloseModule(glb_dlhandle stale)
{
  int s;
  void (*clean)(void);
  lt_dlhandle module;
  module=(lt_dlhandle) stale;
  clean=lt_dlsym(module,"glb_module_clean");
  if(clean) clean();
  s=lt_dlclose(module);
  return s;
}


/* A safe lt_dlsym */
void *glbSymModule(glb_dlhandle module,const char *symbol_name)
{
  const char *error;
  void *generic;
  lt_dlhandle mt;
  mt=(lt_dlhandle) module;
  generic=lt_dlsym(mt,symbol_name);
  error=lt_dlerror();
  if(error) {
    glb_error("Could not resolve symbol in module");
    fprintf(stderr,"\tin resolution of %s failed\n",symbol_name);
    fprintf(stderr,"\t%s\n",error);
    return NULL;}
  return generic;
}

/*****************************************************************************/
/****************** End of general module support ****************************/
/*****************************************************************************/

/* Register the priors as defined in module */
int glbUsePrior(glb_dlhandle module)
{
  int magic,*id;
  double (*f)(const glb_params);
  int (*g)(const glb_params);
  int (*h)(const glb_params);
  const char *error;
  int s=0;

  id=lt_dlsym(module,"glb_module_id");
  magic=*id;

  /* Probing wether it is designed as prior module */
  if(magic!=GLB_PRIOR_MODULE_ID) {glb_error("Not a prior module!");
  return -1;}

  /* getting the correct symbol */
  f=glbSymModule(module,"glb_module_prior");
  g=glbSymModule(module,"glb_module_starting_values");
  h=glbSymModule(module,"glb_module_input_errors");

  /* Register the new prior */
  glbRegisterPriorFunction(f,g,h);

  return 0;
}
