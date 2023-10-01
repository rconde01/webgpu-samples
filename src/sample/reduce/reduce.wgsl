
// Dimension     CUDA             WebGPU                                 Description
// -----------------------------------------------------------------------------------------------------------------------
// n/a           Shared Memory    Workgroup Address Space
// n/a           Global Memory    Storage Address Space
// n/a           __syncThreads    storageBarrier or workgroupBarrier
// n/a           Thread           Invocation
// n/a           Warp             n/a
// 1             warpSize         n/a
// n/a           Thread Block     Workgroup
// 
// 3             blockDim         workgroup_size (wgs)
// 3             numBlocks        num_workgroups (nw)
// 3             blockIdx         workgroup_id (wid)                     position of workgroup within the global grid
//                                workgroup_index (wgi)                  wid.x + wid.y*nw.x + (wid.z)*nw.x*nw.y;
// 3             threadIdx        local_invocation_id (liid)             position of invocation within a workgroup grid
//                                local_invocation_index                 unique index within a workgroup, range [0 -> wgs.x*wgs.y*wgs.z - 1]
//                                                                       liid.x + liid.y*wgs.x + liid.z*(wgs.x*wgs.y)
// 3                              global_invocation_id                   position of invocation within global grid
//                                global_invocation_index                unique index within all invocations 
//                                                                       wgi * wgs + liid

override workgroup_size = 32u;

struct Data {
    data: array<f32>
}

@binding(0) @group(0) var<storage,read> global_input: Data;
@binding(1) @group(0) var<storage,read_write> global_output: Data;

var<workgroup> workgroup_data : array<f32,workgroup_size>;

@compute @workgroup_size(workgroup_size)
fn reduce0(@builtin(local_invocation_id) local_invocation_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_invocation_id.x;
    let i = workgroup_id.x * workgroup_size + local_invocation_id.x;

    if i < arrayLength(&global_input.data) {
        workgroup_data[tid] = global_input.data[i];
    } else {
        workgroup_data[tid] = 0.0;
    }
    workgroupBarrier();

    for (var s = 1u; s < workgroup_size; s *= 2u) {
        if tid % (2u * s) == 0u {
            workgroup_data[tid] += workgroup_data[tid + s];
        }
        workgroupBarrier();
    }

    if tid == 0u {
        global_output.data[workgroup_id.x] = workgroup_data[0];
    }
}

@compute @workgroup_size(workgroup_size)
fn reduce1(@builtin(local_invocation_id) local_invocation_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_invocation_id.x;
    let i = workgroup_id.x * workgroup_size + local_invocation_id.x;

    if i < arrayLength(&global_input.data) {
        workgroup_data[tid] = global_input.data[i];
    } else {
        workgroup_data[tid] = 0.0;
    }

    workgroupBarrier();

    for (var s = 1u; s < workgroup_size; s *= 2u) {
        let index = 2u * s * tid;

        if index < workgroup_size {
            workgroup_data[index] += workgroup_data[index + s];
        }
        workgroupBarrier();
    }

    if tid == 0u {
        global_output.data[workgroup_id.x] = workgroup_data[0];
    }
}

@compute @workgroup_size(workgroup_size)
fn reduce2(@builtin(local_invocation_id) local_invocation_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_invocation_id.x;
    let i = workgroup_id.x * workgroup_size + local_invocation_id.x;

    if i < arrayLength(&global_input.data) {
        workgroup_data[tid] = global_input.data[i];
    } else {
        workgroup_data[tid] = 0.0;
    }

    workgroupBarrier();

    for (var s = workgroup_size / 2u; s > 0u; s >>= 1u) {
        if tid < s {
            workgroup_data[tid] += workgroup_data[tid + s];
        }
        workgroupBarrier();
    }

    if tid == 0u {
        global_output.data[workgroup_id.x] = workgroup_data[0];
    }
}

@compute @workgroup_size(workgroup_size)
fn reduce3(@builtin(local_invocation_id) local_invocation_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_invocation_id.x;
    let i = workgroup_id.x * (workgroup_size * 2u) + local_invocation_id.x;

    var my_sum = 0.0;

    if i < arrayLength(&global_input.data) {
        my_sum = global_input.data[i];
    }

    if i + workgroup_size < arrayLength(&global_input.data) {
        my_sum += global_input.data[i + workgroup_size];
    }

    workgroup_data[tid] = my_sum;
    workgroupBarrier();

    for (var s = workgroup_size / 2u; s > 0u; s >>= 1u) {
        if tid < s {
            my_sum += workgroup_data[tid + s];
            workgroup_data[tid] = my_sum;
        }
        workgroupBarrier();
    }

    if tid == 0u {
        global_output.data[workgroup_id.x] = my_sum;
    }
}