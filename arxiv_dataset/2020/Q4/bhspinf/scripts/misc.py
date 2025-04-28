import numpy as np

'''
Miscellaneous functions.
'''

#====================================================================================================
# Collect the command line keyword arguments (nicked from Jordan Mirocha)
def get_cmd_line_kwargs(argv):

    # Initialize the dictionary of command line keyword arguments
    cmd_line_kwargs = {}

    # Loops through the keyword arguments
    for arg in argv[1:]:
        
        # Split the argument into the variable name (pre) and its value (post)
        #pre, post = arg.split('=')
        pre  = arg.split('=')[0]
        post = arg.split(pre+'=')[-1]

        # Need to do some type-casting
        if post.isdigit():
            cmd_line_kwargs[pre] = int(post)
        elif post.isalpha():
            cmd_line_kwargs[pre] = str(post)
        elif post[0] == '[':
            vals = post[1:-1].split(',')
            #cmd_line_kwargs[pre] = np.array(map(float, vals))
            cmd_line_kwargs[pre] = list(map(float, vals))
        else:
            try:
                cmd_line_kwargs[pre] = float(post)
            except ValueError:
                # Strings with underscores will return False from isalpha()
                cmd_line_kwargs[pre] = str(post)

    return cmd_line_kwargs
    
#====================================================================================================
