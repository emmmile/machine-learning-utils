TEMPLATE = app
CONFIG += console
CONFIG += qt

SOURCES += main.cpp

HEADERS += \
    random.h \
    vect.hpp \
    particle.hpp \
    pso.hpp \
    ackley.hpp

LIBS += -s
QMAKE_CXXFLAGS += -O3 -ffast-math -std=c++0x -funroll-loops -mfpmath=sse -msse -msse2 -msse3 -mssse3 -march=corei7
QMAKE_CXXFLAGS -= -O2 -g -march=x86-64 -mtune=generic

