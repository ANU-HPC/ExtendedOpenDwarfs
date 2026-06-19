TOP_LEVEL ?= ../../..
include $(TOP_LEVEL)/makefile_defs.mk

TARGET := gemnoui-opencl
BUILD_DIR := build/opencl

C_SRCS := \
	../dump_vertices.c \
	../floating_centers.c \
	../radix_sort.c \
	../read_pqr.c \
	../vector_math.c \
	../write_grid.c \
	../extrapolate_bonds.c \
	../populate_stats.c \
	../read_msms.c \
	../run_msms.c \
	../write_avs.c \
	../write_xyzr.c

CXX_SRCS := \
	calculate_potential.cpp \
	../check_cmdline.cpp \
	../estimate_a.cpp \
	../gem_no_ui.cpp \
	../open_pqr_run_msms.cpp

OBJS := \
	$(BUILD_DIR)/dump_vertices.o \
	$(BUILD_DIR)/floating_centers.o \
	$(BUILD_DIR)/radix_sort.o \
	$(BUILD_DIR)/read_pqr.o \
	$(BUILD_DIR)/vector_math.o \
	$(BUILD_DIR)/write_grid.o \
	$(BUILD_DIR)/extrapolate_bonds.o \
	$(BUILD_DIR)/populate_stats.o \
	$(BUILD_DIR)/read_msms.o \
	$(BUILD_DIR)/run_msms.o \
	$(BUILD_DIR)/write_avs.o \
	$(BUILD_DIR)/write_xyzr.o \
	$(BUILD_DIR)/calculate_potential.o \
	$(BUILD_DIR)/check_cmdline.o \
	$(BUILD_DIR)/estimate_a.o \
	$(BUILD_DIR)/gem_no_ui.o \
	$(BUILD_DIR)/open_pqr_run_msms.o \
	$(BUILD_DIR)/common_args.o \
	$(BUILD_DIR)/opts.o \
	$(BUILD_DIR)/rdtsc.o

KERNELS := \
	calculate_potential.cl

INCLUDES := \
	-I. \
	-I.. \
	-I../include \
	-I../include/visualize/dialogs \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(TOP_LEVEL)/opts \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

COMMON_CPPFLAGS := $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES) -DNO_UI

.PHONY: all clean run run-$(TARGET)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) \
		$(CXXFLAGS) \
		-o $@ \
		$^ \
		$(LDFLAGS) \
		$(OPENCL_LDFLAGS) \
		$(LDLIBS) \
		$(OPENCL_LDLIBS) \
		-lm

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: ../%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: ../%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/calculate_potential.o: calculate_potential.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/common_args.o: $(OCD_COMMON_ARGS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/opts.o: $(OCD_OPTS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/rdtsc.o: $(OCD_RDTSC_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

run: run-$(TARGET)

run-$(TARGET): $(TARGET)
	mkdir -p $(TOP_LEVEL)/results
	cp $(KERNELS) $(TOP_LEVEL)/results/
	cd $(TOP_LEVEL)/results && $(RUN_ENV) $(CURDIR)/$(TARGET) $(ARGS) $(N)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET)
