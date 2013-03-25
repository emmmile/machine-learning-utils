


env = Environment(
  CCFLAGS = [
    '-Wall', '-O3', '-std=c++11',
    #'-ffast-math', '-funroll-loops', '-pipe', '-Wno-deprecated', '-msse2',
    #'-mfpmath=sse', '-fomit-frame-pointer'
  ],

  CPPPATH=['#./ml','#./math'],
  LIBS=[],
  LIBPATH=[],
  CPPDEFINES=[],
  LINKFLAGS=['-s', '-Os']
)

Export('env')



SConscript(['test/SConscript'])