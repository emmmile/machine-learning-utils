


env = Environment(
  CCFLAGS = [
    '-Wall', '-Werror', '-std=c++11',
    '-O3', '-march=native', '-ffast-math', '-funroll-loops', '-pipe',
    #'-msse', '-msse2', '-msse3', '-mssse3', '-msse4.1', '-msse4.2', '-mfpmath=sse',
  ],

  CPPPATH=['#./ml','#./math'],
  LIBS=[],
  LIBPATH=[],
  CPPDEFINES=[],
  LINKFLAGS=['-s', '-Os']
)

Export('env')



SConscript(['test/SConscript'])
