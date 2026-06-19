TOP_LEVEL ?= ../../..
include $(TOP_LEVEL)/makefile_defs.mk

TARGET := kmeans-opencl
PROFILE_TARGET := kmeans_profiling_outer_loop-opencl
BUILD_DIR := build/opencl

SRCS := \
	../kmeans.c \
	../cluster.c \
	../getopt.c \
	../kmeans_clustering.c \
	kmeans.cpp \
	../rmse.c \
	$(OCD_COMMON_ARGS_SRC) \
	$(OCD_OPTS_SRC) \
	$(OCD_RDTSC_SRC)

OBJS := \
	$(BUILD_DIR)/kmeans_main.o \
	$(BUILD_DIR)/cluster.o \
	$(BUILD_DIR)/getopt.o \
	$(BUILD_DIR)/kmeans_clustering.o \
	$(BUILD_DIR)/rmse.o \
	$(BUILD_DIR)/common_args.o \
	$(BUILD_DIR)/opts.o \
	$(BUILD_DIR)/rdtsc.o \
	$(BUILD_DIR)/kmeans_opencl_backend.o

PROFILE_OBJS := \
	$(BUILD_DIR)/profile-kmeans_main.o \
	$(BUILD_DIR)/profile-cluster.o \
	$(BUILD_DIR)/profile-getopt.o \
	$(BUILD_DIR)/profile-kmeans_clustering.o \
	$(BUILD_DIR)/profile-rmse.o \
	$(BUILD_DIR)/profile-common_args.o \
	$(BUILD_DIR)/profile-opts.o \
	$(BUILD_DIR)/profile-rdtsc.o \
	$(BUILD_DIR)/profile-kmeans_opencl_backend.o

KERNELS := \
	kmeans_kernel.cl \
	kmeans_kernel_opt_gpu.cl

INCLUDES := \
	-I. \
	-I.. \
	-I$(TOP_LEVEL) \
	-I$(TOP_LEVEL)/include \
	-I$(TOP_LEVEL)/opts \
	-I$(LSB_INSTALL_ROOT)/include \
	-I$(LSB_INSTALL_ROOT)/lib/include

COMMON_CPPFLAGS := $(CPPFLAGS) $(OPENCL_CPPFLAGS) $(INCLUDES)

.PHONY: all clean run run-$(TARGET) run-$(PROFILE_TARGET)

all: $(TARGET) $(PROFILE_TARGET)

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

$(PROFILE_TARGET): $(PROFILE_OBJS)
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

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/kmeans_main.o: ../kmeans.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/kmeans_opencl_backend.o: kmeans.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/common_args.o: $(OCD_COMMON_ARGS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/opts.o: $(OCD_OPTS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/rdtsc.o: $(OCD_RDTSC_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/profile-%.o: ../%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -DPROFILE_OUTER_LOOP -c $< -o $@

$(BUILD_DIR)/profile-%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_CPPFLAGS) -DPROFILE_OUTER_LOOP -c $< -o $@

$(BUILD_DIR)/profile-kmeans_main.o: ../kmeans.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -DPROFILE_OUTER_LOOP -c $< -o $@

$(BUILD_DIR)/profile-kmeans_opencl_backend.o: kmeans.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(COMMON_CPPFLAGS) -DPROFILE_OUTER_LOOP -c $< -o $@

$(BUILD_DIR)/profile-common_args.o: $(OCD_COMMON_ARGS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -DPROFILE_OUTER_LOOP -c $< -o $@

$(BUILD_DIR)/profile-opts.o: $(OCD_OPTS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -DPROFILE_OUTER_LOOP -c $< -o $@

$(BUILD_DIR)/profile-rdtsc.o: $(OCD_RDTSC_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(COMMON_CPPFLAGS) -DPROFILE_OUTER_LOOP -c $< -o $@

run: run-$(TARGET)

run-$(TARGET): $(TARGET)
	mkdir -p $(TOP_LEVEL)/results
	cp $(KERNELS) $(TOP_LEVEL)/results/
	cd $(TOP_LEVEL)/results && $(RUN_ENV) $(CURDIR)/$(TARGET) $(ARGS)

run-$(PROFILE_TARGET): $(PROFILE_TARGET)
	mkdir -p $(TOP_LEVEL)/results
	cp $(KERNELS) $(TOP_LEVEL)/results/
	cd $(TOP_LEVEL)/results && $(RUN_ENV) $(CURDIR)/$(PROFILE_TARGET) $(ARGS)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET) $(PROFILE_TARGET)
