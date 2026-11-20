// Reads the imported buffer through a pointer stored in memory, so the pointer is not a
// kernel argument and the buffer is only reachable through the driver's indirect-access
// residency path (zeKernelSetIndirectAccess).
kernel void read_through_table(global const ulong *table, global uchar *out, uint stride) {
    size_t i = get_global_id(0);
    global const uchar *p = (global const uchar *)table[0];
    out[i] = p[i * stride];
}
