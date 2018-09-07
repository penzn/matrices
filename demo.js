function init(arr, size) {
  for (let i = 0; i < (size*size); ++i) {
    arr[i] = i % size;
  }
}

function verify(arr, size) {
  for (let i = 0; i < (size*size); ++i) {
    if (arr[i] !== i % size) {
      return false;
    }
  }
  return true;
}

function print_4(arr, off) {
  print("2x2 data: ");
  print(arr[off]     + " " + arr[off + 1]);
  print(arr[off + 2] + " " + arr[off + 3]);
}

function print_16(arr) {
  print("4x4 data: ");
  print(arr[0] + " " + arr[1] + " " + arr[2] + " " + arr[3]);
  print(arr[4] + " " + arr[5] + " " + arr[6] + " " + arr[7]);
  print(arr[8] + " " + arr[9] + " " + arr[10] + " " + arr[11]);
  print(arr[12] + " " + arr[13] + " " + arr[14] + " " + arr[15]);
}

// Calculate GFLOPS for running time (ms) and matrix size
function getGFLOPS(ms, N) {
  // For every scalar there is a a multiplication and an addition, and there
  // are N^3 visits of scalars
  let ops = N*N*N*2;
  return Math.round(ops / t / 10000) / 100;
}

const N = 3000; // Matrix size
let pages = Math.ceil(3 * N * N / 16384.0);
print("Need " + pages + " pages");
const memObj = new WebAssembly.Memory({initial:pages});
const module = new WebAssembly.Module(readbuffer('matrices.wasm'));
const instance = new WebAssembly.Instance(module, { "dummy" : { "memory" : memObj } }).exports;
let data = new Float32Array (memObj.buffer);

print("Matrix size is " + N);

init(data, N);
// A*A
for (let i = 0; i < (N * N); ++i) {
  data[N * N + i] = data[i];
}

instance["transpose_f32"](4*N*N, N); // Second argument to column-major order

var tStart = Date.now();
instance["multiply_f32"](0, 4*N*N, 8*N*N, N);
var tEnd = Date.now();
let t = tEnd - tStart;
print("Scalar multiplication took " + t + " ms at " + getGFLOPS(t, N) + " SP GFLOPS.");

var tStart = Date.now();
instance["multiply_f32_simd"](0, 4*N*N, 8*N*N, N);
var tEnd = Date.now();
t = tEnd - tStart;

print("SIMD multiplication took " + t + " ms at " + getGFLOPS(t, N) + " SP GFLOPS.");

