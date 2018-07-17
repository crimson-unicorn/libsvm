CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: svm-train svm-predict svm-scale incocsvm-train

lib: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} svm.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.c svm.o
	$(CXX) $(CFLAGS) svm-predict.c svm.o -o svm-predict -lm
incocsvm-predict: incocsvm-predict.c svm.o
	$(CXX) $(CFLAGS) incocsvm-predict.c svm.o -o incocsvm-predict -lm
svm-train: svm-train.c svm.o
	$(CXX) $(CFLAGS) svm-train.c svm.o -o svm-train -lm
incocsvm-train: incocsvm-train.c svm.o
	$(CXX) $(CFLAGS) incocsvm-train.c svm.o -o incocsvm-train -lm
svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp
clean:
	rm -f *~ svm.o svm-train svm-predict svm-scale incocsvm-train libsvm.so.$(SHVER)
