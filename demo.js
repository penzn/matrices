// Convert a pair of integers to a float in a way that is unique for each distinct pair
function p2f(x, y) {
  var d;
  for (d = 10; Math.floor(y/d); d = d * 10);
  return x + y / d;
}

function init_f32(arr, size) {
  for (let i = 0; i < (size*size); ++i) {
    arr[i] = i % size + 0.5;
  }
}

function init_f32_unique(arr, size) {
  for (let i = 0; i < size; ++i) {
    for (let j = 0; j < size; ++j) {
      arr[i * size + j] = p2f(i,j);
    }
  }
}

function verify_f32_unique(arr, size) {
  for (let i = 0; i < size; ++i) {
    for (let j = 0; j < size; ++j) {
      if (arr[i * size + j] !== p2f(i,j)) {
        print("Expected: ", p2f(i,j));
        print("But got: ", arr[i*size+j]);
        return false;
      }
    }
  }
  return true;
}

function init_i32(arr, size) {
  for (let i = 0; i < (size*size); ++i) {
    arr[i] = i % size + 1;
  }
}

function verify_f32(arr, size) {
  for (let i = 0; i < (size*size); ++i) {
    if (arr[i] !== (i % size + 0.5)) {
      return false;
    }
  }
  return true;
}

function print_4(arr, off) {
  print("2x2: ");
  print(arr[off]     + " " + arr[off + 1]);
  print(arr[off + 2] + " " + arr[off + 3]);
}

function print_16(arr) {
  print("4x4: ");
  print(arr[0] + " " + arr[1] + " " + arr[2] + " " + arr[3]);
  print(arr[4] + " " + arr[5] + " " + arr[6] + " " + arr[7]);
  print(arr[8] + " " + arr[9] + " " + arr[10] + " " + arr[11]);
  print(arr[12] + " " + arr[13] + " " + arr[14] + " " + arr[15]);
}

function print_arr(arr, size) {
  print(size + "x" + size + ": ");
  for (let i = 0; i < size; ++i) {
    var line = "";
    for (let j = 0; j < size; ++j) {
      line = line + arr[size * i + j] + " ";
    }
    print(line);
  }
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
let f32_data = new Float32Array (memObj.buffer);
let i32_data = new Int32Array (memObj.buffer);

print("Matrix size is " + N);

init_f32(f32_data, N);

// A*A
for (let i = 0; i < (N * N); ++i) {
  f32_data[N * N + i] = f32_data[i];
}

instance["transpose_32"](4*N*N, N); // Second argument to column-major order

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

// Transpose test
var tStart = Date.now();
instance["test_transpose_32"](0, N, 500);
var tEnd = Date.now();
t = tEnd - tStart;
print("Transpose took " + t + " ms");
if (verify_f32(f32_data, N)) {
  print("Verfication passed");
} else {
  print("Verfication failed");
}

init_i32(i32_data, N);
// A*A
for (let i = 0; i < (N * N); ++i) {
  i32_data[N * N + i] = i32_data[i];
}

instance["transpose_32"](4*N*N, N); // Second argument to column-major order

var tStart = Date.now();
instance["multiply_i32"](0, 4*N*N, 8*N*N, N);
var tEnd = Date.now();
t = tEnd - tStart;
print("Scalar integer multiplication took " + t + " ms");

var tStart = Date.now();
instance["multiply_i32_simd"](0, 4*N*N, 8*N*N, N);
var tEnd = Date.now();
t = tEnd - tStart;
print("SIMD integer multiplication took " + t + " ms");

