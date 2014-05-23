

# http://gcc.gnu.org/bugzilla/show_bug.cgi?id=45483
# http://en.chys.info/2010/04/what-exactly-marchnative-means/
# from gcc 4.5 -march=native wrongly detects some intel cpus, including mine.
# so better using -mtune=generic or -mtune=native

env = Environment(
  CCFLAGS = [
    '-Wall', '-Werror', '-std=c++11',
    '-O3', '-mtune=native', '-ffast-math', '-funroll-loops', '-pipe',
    #'-msse', '-msse2', '-msse3', '-mssse3', '-msse4.1', '-msse4.2', '-mfpmath=sse',
  ],

  CPPPATH=['#./ml','#./math'],
  LIBS=[],
  LIBPATH=[],
  CPPDEFINES=[],
  LINKFLAGS=[
    '-Os'
    #'-g'
  ]
)

Export('env')



SConscript(['test/SConscript'])
