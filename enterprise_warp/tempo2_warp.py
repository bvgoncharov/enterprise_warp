import subprocess
import sys

def get_tempo2_prediction(par,tim,configuration,output_file,execute='tempo2'):
  """
  Runs tempo2 and returns noise reconstruction in a file.
  Plugin "general2" is used.
  Configuration is a string in a string, example:
    '"{bat}\t{freq}\t{post}\t{err}\t{posttn}\t{tndm}\t{tnrn}\n"'.
  Kwarg execute: full path to tempo2 executable file, if other user's tempo2
    executable is sourced and aliased.
  See tempo2 manual online for more information.
  Output can be loaded with numpy.loadtxt().
  """
  
  command = [execute,'-output','general2','-f',par,tim,'-s',configuration]

  try:
    result = subprocess.check_output(command)
  except subprocess.CalledProcessError as t2_run_exception:
    try:
      # If the error is from "Too many TOAs" of tempo2
      print('get_tempo2_prediction: handing tempo2 exception')
      command.append('-nobs')
      command.append('1000000')
      result = subprocess.check_output(command)
    except subprocess.CalledProcessError as ee:
      print('Unknown error when running tempo2. Exiting.')
      sys.exit(1)
  result = result.decode("utf-8") # bytes to string
  # Remove output junk before and after the actual output:
  result = result.partition('Starting general2 plugin')[2]
  result = result.partition('Finished general2 plugin')[0]
  result = result.replace('"','') # remove quotation marks
  
  with open(output_file, "w") as output: output.write(result)

