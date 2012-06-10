TEMPLATE = app
CONFIG += console
CONFIG += qt

SOURCES += main.cpp

HEADERS += \
    vect.hpp \
    particle.hpp \
    pso.hpp \
    ackley.hpp \
    random.hpp

OTHER_FILES += \
    Sconstruct
