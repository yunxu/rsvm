# Make command to use for dependencies
MAKE=gmake
RM=rm
MKDIR=mkdir
UNAME=uname
HOSTNAME=hostname
AWK=awk
# -----Begin user-editable area-----
CC=g++
# -----End user-editable area-----

# If no configuration is specified, "Debug" will be used
# uncomment if compile in release mode
# option could choose between static and share
CFG=Release
CFG_OPTION=static

ifndef CFG
CFG=Debug
endif

#
# Configuration: Release
# put every obj file into Debug directory rather than the Source file directory
#
ifeq "$(CFG)" "Release"
OUTDIR=Release

PLATFORM=$(UNAME) -i
HOST=$(HOSTNAME) -s 
ifeq "$(shell $(PLATFORM))" "x86_64"
OUTFILE=$(OUTDIR)/rdsvm_learn_64
CFG_LIB=-L/home/yunxu/tools/lapack-3.1.1_64 -llapack_LINUX -lblas_LINUX -lg2c -lm
else
OUTFILE=$(OUTDIR)/rdsvm_learn_32
CFG_LIB=-L/home/yunxu/tools/lapack-3.1.1_32 -llapack_LINUX -lblas_LINUX -lg2c -lm
endif

ifeq "$(shell $(HOST))" "ubeda"
CFG_LIB=-llapack -lblas -lgfortran -lm
endif

ifeq "$(shell $(HOST))" "master-fw"
CFG_LIB=-llapack -lblas -lgfortran -lm
endif

INC_DIR1=./
CFG_INC=-I$(INC_DIR1) 
CFG_OBJ1:=$(addprefix $(OUTDIR)/,$(notdir $(patsubst %.cpp,%.o,$(wildcard $(addprefix $(INC_DIR1)/,*.cpp)))))
#COMMON_OBJ=$(OUTDIR)/rdsvm_learn.o 
OBJ=$(COMMON_OBJ) $(CFG_OBJ1)
ALL_OBJ=$(OUTDIR)/rdsvm_learn.o 

COMPILE=$(CC) -c		    -o "$(OUTDIR)/$(*F).o"	 $(CFG_INC) "$<"


ifeq "$(CFG_OPTION)" "static"
 LINK=$(CC) -O3 -static		-o "$(OUTFILE)"	 $(OBJ) $(CFG_LIB) 
else
 LINK=$(CC) -O3 		-o "$(OUTFILE)"	 $(OBJ) $(CFG_LIB)
endif

# Do not need share library
#LINK=g++ -O3 -static		-o "$(OUTFILE)"	 $(OBJ) $(CFG_LIB) 
 
# Pattern rules
$(OUTDIR)/%.o : %.cpp 
	$(COMPILE)


$(OUTDIR)/%.o : $(INC_DIR1)/%.cpp
	$(COMPILE)

# Build rules
all: $(OUTFILE)

$(OUTFILE): $(OUTDIR)  $(OBJ)
	@echo $(shell $(HOST))
	$(LINK)
	cp $(OUTFILE) ./

$(OUTDIR):
	$(MKDIR) -p "$(OUTDIR)"

# Rebuild this project
rebuild: cleanall all

# Clean this project
clean:
	$(RM) -f $(OUTFILE)
	$(RM) -f $(OBJ)
# Clean this project and all dependencies
cleanall: clean
endif









