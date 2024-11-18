# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# 3.1 and 3.2 Diagnostic Output

```
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/georgehuang2020/Repositories/Cornell/cs
5781/mod3-gzfh24/minitorch/fast_ops.py (164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/georgehuang2020/Repositories/Cornell/cs 5781/mod3-gzfh24/minitorch/fast_ops.py (164)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        # TODO: Implement for Task 3.1.                                      |
        if np.array_equal(in_strides, out_strides) and np.array_equal(       |
            in_shape, out_shape                                              |
        ):                                                                   |
            for i in prange(len(out)):---------------------------------------| #0
                out[i] = fn(in_storage[i])  # type: ignore                   |
        else:                                                                |
            for i in prange(len(out)):---------------------------------------| #1
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)        |
                in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)         |
                to_index(i, out_shape, out_index)                            |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                o = index_to_position(out_index, out_strides)                |
                j = index_to_position(in_index, in_strides)                  |
                out[o] = fn(in_storage[j])                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/georgehuang2020/Repositories/Cornell/cs
5781/mod3-gzfh24/minitorch/fast_ops.py (180) is hoisted out of the parallel loop
 labelled #1 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/georgehuang2020/Repositories/Cornell/cs
5781/mod3-gzfh24/minitorch/fast_ops.py (181) is hoisted out of the parallel loop
 labelled #1 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/georgehuang2020/Repositories/Cornell/cs
5781/mod3-gzfh24/minitorch/fast_ops.py (214)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/georgehuang2020/Repositories/Cornell/cs 5781/mod3-gzfh24/minitorch/fast_ops.py (214)
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              |
        out: Storage,                                                      |
        out_shape: Shape,                                                  |
        out_strides: Strides,                                              |
        a_storage: Storage,                                                |
        a_shape: Shape,                                                    |
        a_strides: Strides,                                                |
        b_storage: Storage,                                                |
        b_shape: Shape,                                                    |
        b_strides: Strides,                                                |
    ) -> None:                                                             |
        # TODO: Implement for Task 3.1.                                    |
        if (                                                               |
            np.array_equal(a_strides, b_strides)                           |
            and np.array_equal(a_strides, out_strides)                     |
            and np.array_equal(a_shape, b_shape)                           |
            and np.array_equal(a_shape, out_shape)                         |
        ):                                                                 |
            for i in prange(len(out)):-------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])  # type: ignore    |
        else:                                                              |
            for i in prange(len(out)):-------------------------------------| #3
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)      |
                a_index: Index = np.empty(MAX_DIMS, dtype=np.int32)        |
                b_index: Index = np.empty(MAX_DIMS, dtype=np.int32)        |
                to_index(i, out_shape, out_index)                          |
                broadcast_index(out_index, out_shape, a_shape, a_index)    |
                broadcast_index(out_index, out_shape, b_shape, b_index)    |
                o = index_to_position(out_index, out_strides)              |
                a_pos = index_to_position(a_index, a_strides)              |
                b_pos = index_to_position(b_index, b_strides)              |
                out[o] = fn(a_storage[a_pos], b_storage[b_pos])            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/georgehuang2020/Repositories/Cornell/cs
5781/mod3-gzfh24/minitorch/fast_ops.py (236) is hoisted out of the parallel loop
 labelled #3 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/georgehuang2020/Repositories/Cornell/cs
5781/mod3-gzfh24/minitorch/fast_ops.py (237) is hoisted out of the parallel loop
 labelled #3 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: a_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/georgehuang2020/Repositories/Cornell/cs
5781/mod3-gzfh24/minitorch/fast_ops.py (238) is hoisted out of the parallel loop
 labelled #3 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: b_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/georgehuang2020/Repositories/Cornell/cs
5781/mod3-gzfh24/minitorch/fast_ops.py (271)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/georgehuang2020/Repositories/Cornell/cs 5781/mod3-gzfh24/minitorch/fast_ops.py (271)
---------------------------------------------------------------------|loop #ID
    def _reduce(                                                     |
        out: Storage,                                                |
        out_shape: Shape,                                            |
        out_strides: Strides,                                        |
        a_storage: Storage,                                          |
        a_shape: Shape,                                              |
        a_strides: Strides,                                          |
        reduce_dim: int,                                             |
    ) -> None:                                                       |
        # TODO: Implement for Task 3.1.                              |
        for i in prange(len(out)):-----------------------------------| #4
            out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)    |
            reduce_size = a_shape[reduce_dim]                        |
            to_index(i, out_shape, out_index)                        |
            o = index_to_position(out_index, out_strides)            |
            j = index_to_position(out_index, a_strides)              |
            delta = a_strides[reduce_dim]                            |
            acc = out[o]                                             |
            for _ in range(reduce_size):                             |
                acc = fn(acc, a_storage[j])                          |
                j += delta                                           |
            out[o] = acc                                             |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/georgehuang2020/Repositories/Cornell/cs
5781/mod3-gzfh24/minitorch/fast_ops.py (282) is hoisted out of the parallel loop
 labelled #4 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/georgehuang2020/Repositories/Cornell/cs
5781/mod3-gzfh24/minitorch/fast_ops.py (297)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/georgehuang2020/Repositories/Cornell/cs 5781/mod3-gzfh24/minitorch/fast_ops.py (297)
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                |
    out: Storage,                                                                           |
    out_shape: Shape,                                                                       |
    out_strides: Strides,                                                                   |
    a_storage: Storage,                                                                     |
    a_shape: Shape,                                                                         |
    a_strides: Strides,                                                                     |
    b_storage: Storage,                                                                     |
    b_shape: Shape,                                                                         |
    b_strides: Strides,                                                                     |
) -> None:                                                                                  |
    """NUMBA tensor matrix multiply function.                                               |
                                                                                            |
    Should work for any tensor shapes that broadcast as long as                             |
                                                                                            |
    ```                                                                                     |
    assert a_shape[-1] == b_shape[-2]                                                       |
    ```                                                                                     |
                                                                                            |
    Optimizations:                                                                          |
                                                                                            |
    * Outer loop in parallel                                                                |
    * No index buffers or function calls                                                    |
    * Inner loop should have no global writes, 1 multiply.                                  |
                                                                                            |
                                                                                            |
    Args:                                                                                   |
    ----                                                                                    |
        out (Storage): storage for `out` tensor                                             |
        out_shape (Shape): shape for `out` tensor                                           |
        out_strides (Strides): strides for `out` tensor                                     |
        a_storage (Storage): storage for `a` tensor                                         |
        a_shape (Shape): shape for `a` tensor                                               |
        a_strides (Strides): strides for `a` tensor                                         |
        b_storage (Storage): storage for `b` tensor                                         |
        b_shape (Shape): shape for `b` tensor                                               |
        b_strides (Strides): strides for `b` tensor                                         |
                                                                                            |
    Returns:                                                                                |
    -------                                                                                 |
        None : Fills in `out`                                                               |
                                                                                            |
    """                                                                                     |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  |
                                                                                            |
    for i in prange(out_shape[0]):----------------------------------------------------------| #5
        for j in range(out_shape[-2]):                                                      |
            for k in range(out_shape[-1]):                                                  |
                a_pos = i * a_batch_stride + j * a_strides[-2]                              |
                b_pos = i * b_batch_stride + k * b_strides[-1]                              |
                acc = 0.0                                                                   |
                for _ in range(a_shape[-1]):                                                |
                    acc += a_storage[a_pos] * b_storage[b_pos]                              |
                    a_pos += a_strides[-1]                                                  |
                    b_pos += b_strides[-2]                                                  |
                out_pos = i * out_strides[0] + j * out_strides[-2] + k * out_strides[-1]    |
                out[out_pos] = acc                                                          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# 3.5

## CPU

50 Points, 100 Hidden Layers, 0.05 Learning Rate

Simple:
```
Epoch 0, Loss 4.797834552557831, Correct 31, Time 14.72060489654541 seconds
Epoch 10, Loss 0.7331655631251182, Correct 49, Time 0.10839509963989258 seconds
Epoch 20, Loss 0.4027477350545535, Correct 50, Time 0.10894036293029785 seconds
Epoch 30, Loss 0.4172670530974163, Correct 50, Time 0.11162042617797852 seconds
Epoch 40, Loss 0.7922193446430601, Correct 50, Time 0.11085867881774902 seconds
Epoch 50, Loss 0.3543053604380908, Correct 50, Time 0.10981464385986328 seconds
Epoch 60, Loss 0.17653241115741028, Correct 50, Time 0.1104576587677002 seconds
Epoch 70, Loss 0.6806783062571256, Correct 50, Time 0.11122584342956543 seconds
Epoch 80, Loss 0.3878730454405987, Correct 50, Time 0.11121225357055664 seconds
Epoch 90, Loss 0.5094155311565558, Correct 50, Time 0.11974930763244629 seconds
Epoch 100, Loss 0.5719250528405918, Correct 50, Time 0.2415931224822998 seconds
Epoch 110, Loss 0.8503446001038358, Correct 50, Time 0.11009883880615234 seconds
Epoch 120, Loss 0.12025076264738159, Correct 50, Time 0.1100623607635498 seconds
Epoch 130, Loss 0.5572287020498835, Correct 50, Time 0.11265087127685547 seconds
Epoch 140, Loss 1.015569888649194, Correct 50, Time 0.10779881477355957 seconds
Epoch 150, Loss 0.3015955339047552, Correct 50, Time 0.11968135833740234 seconds
Epoch 160, Loss 0.21875123913031552, Correct 50, Time 0.1129140853881836 seconds
Epoch 170, Loss 0.004780151804991629, Correct 50, Time 0.10973381996154785 seconds
Epoch 180, Loss 0.023837674719842657, Correct 50, Time 0.10965299606323242 seconds
Epoch 190, Loss 0.22113060898939862, Correct 50, Time 0.10788941383361816 seconds
Epoch 200, Loss 0.026997029599007374, Correct 50, Time 0.18093299865722656 seconds
Epoch 210, Loss 0.8154215198772905, Correct 50, Time 0.10976767539978027 seconds
Epoch 220, Loss 0.0025217717666818538, Correct 50, Time 0.1112370491027832 seconds
Epoch 230, Loss 0.3834410313449468, Correct 50, Time 0.10965466499328613 seconds
Epoch 240, Loss 0.14315906464790698, Correct 50, Time 0.11931276321411133 seconds
Epoch 250, Loss 0.2783725617947184, Correct 50, Time 0.1122891902923584 seconds
Epoch 260, Loss 0.024057123323196768, Correct 50, Time 0.1154475212097168 seconds
Epoch 270, Loss 0.7235941654442023, Correct 50, Time 0.11268401145935059 seconds
Epoch 280, Loss 0.03789729314711075, Correct 50, Time 0.11087203025817871 seconds
Epoch 290, Loss 0.13745352419212495, Correct 50, Time 0.10862851142883301 seconds
Epoch 300, Loss 0.021800989357888427, Correct 50, Time 0.2216024398803711 seconds
Epoch 310, Loss 0.08084119082504114, Correct 50, Time 0.1112518310546875 seconds
Epoch 320, Loss 0.12493229794292303, Correct 50, Time 0.12447762489318848 seconds
Epoch 330, Loss 0.7350065262097699, Correct 50, Time 0.11186695098876953 seconds
Epoch 340, Loss 0.5319449056562693, Correct 50, Time 0.10866475105285645 seconds
Epoch 350, Loss 0.16043296868631804, Correct 50, Time 0.11030936241149902 seconds
Epoch 360, Loss 0.06502830795293259, Correct 50, Time 0.10863876342773438 seconds
Epoch 370, Loss 0.7816902366381814, Correct 50, Time 0.11105513572692871 seconds
Epoch 380, Loss 0.695085283286925, Correct 50, Time 0.10937643051147461 seconds
Epoch 390, Loss 0.14313730307420028, Correct 50, Time 0.11672544479370117 seconds
Epoch 400, Loss 0.0068636892620963855, Correct 50, Time 0.21210813522338867 seconds
Epoch 410, Loss 0.005223928735646833, Correct 50, Time 0.11103701591491699 seconds
Epoch 420, Loss 0.0027399610204792515, Correct 50, Time 0.11361360549926758 seconds
Epoch 430, Loss 0.0006485466149100242, Correct 50, Time 0.10899090766906738 seconds
Epoch 440, Loss 0.037392285248913616, Correct 50, Time 0.11064839363098145 seconds
Epoch 450, Loss 0.4656421328990091, Correct 50, Time 0.11592555046081543 seconds
Epoch 460, Loss 0.058512034879396206, Correct 50, Time 0.11067891120910645 seconds
Epoch 470, Loss 0.07140975180156543, Correct 50, Time 0.10875272750854492 seconds
Epoch 480, Loss 0.7889583450735346, Correct 50, Time 0.10908675193786621 seconds
Epoch 490, Loss 0.07884010827454896, Correct 50, Time 0.10940313339233398 seconds
Average Time Per Epoch: 0.12570105430358397 s
```

Split:
```
Epoch 0, Loss 7.7003080116553955, Correct 19, Time 14.402568578720093 seconds
Epoch 10, Loss 6.419580441982268, Correct 39, Time 0.10736823081970215 seconds
Epoch 20, Loss 5.562671244885048, Correct 40, Time 0.19881033897399902 seconds
Epoch 30, Loss 5.543321606753514, Correct 43, Time 0.13498187065124512 seconds
Epoch 40, Loss 3.970722033357078, Correct 44, Time 0.11057615280151367 seconds
Epoch 50, Loss 2.1497246482734127, Correct 44, Time 0.10727524757385254 seconds
Epoch 60, Loss 2.491834064823838, Correct 49, Time 0.10898685455322266 seconds
Epoch 70, Loss 3.35654902675607, Correct 47, Time 0.1086270809173584 seconds
Epoch 80, Loss 1.4828607588860843, Correct 47, Time 0.10882306098937988 seconds
Epoch 90, Loss 3.6783634700475596, Correct 46, Time 0.10706520080566406 seconds
Epoch 100, Loss 1.8255474696005665, Correct 49, Time 0.10769891738891602 seconds
Epoch 110, Loss 0.9046225066303399, Correct 48, Time 0.11170172691345215 seconds
Epoch 120, Loss 1.5718232759086355, Correct 46, Time 0.20421862602233887 seconds
Epoch 130, Loss 1.0221295939476782, Correct 49, Time 0.24544811248779297 seconds
Epoch 140, Loss 0.6271635165270344, Correct 49, Time 0.10858869552612305 seconds
Epoch 150, Loss 0.6734553115218632, Correct 49, Time 0.11069297790527344 seconds
Epoch 160, Loss 1.7724174814066482, Correct 50, Time 0.10895466804504395 seconds
Epoch 170, Loss 0.678933983276971, Correct 48, Time 0.10767984390258789 seconds
Epoch 180, Loss 1.0688059281347624, Correct 49, Time 0.10972452163696289 seconds
Epoch 190, Loss 1.7187148443502325, Correct 49, Time 0.1111001968383789 seconds
Epoch 200, Loss 0.797677434437857, Correct 48, Time 0.1092538833618164 seconds
Epoch 210, Loss 1.4266170577318138, Correct 50, Time 0.10723519325256348 seconds
Epoch 220, Loss 0.43660805374618167, Correct 48, Time 0.18929171562194824 seconds
Epoch 230, Loss 2.6933913564353684, Correct 49, Time 0.2553117275238037 seconds
Epoch 240, Loss 2.364408031935858, Correct 48, Time 0.1120002269744873 seconds
Epoch 250, Loss 3.196775813398183, Correct 49, Time 0.10995912551879883 seconds
Epoch 260, Loss 1.0142318066067924, Correct 49, Time 0.10930585861206055 seconds
Epoch 270, Loss 1.1685530393453543, Correct 48, Time 0.10742640495300293 seconds
Epoch 280, Loss 0.917429560344019, Correct 48, Time 0.1069037914276123 seconds
Epoch 290, Loss 1.5878430090085356, Correct 48, Time 0.10889005661010742 seconds
Epoch 300, Loss 3.250316432323241, Correct 48, Time 0.1070408821105957 seconds
Epoch 310, Loss 2.2805614665503464, Correct 49, Time 0.10908102989196777 seconds
Epoch 320, Loss 1.477414891757925, Correct 48, Time 0.11295151710510254 seconds
Epoch 330, Loss 0.8149251814552207, Correct 49, Time 0.16804933547973633 seconds
Epoch 340, Loss 0.5373266771821735, Correct 49, Time 0.11397171020507812 seconds
Epoch 350, Loss 0.29982807514210524, Correct 49, Time 0.10855555534362793 seconds
Epoch 360, Loss 0.24292570343080516, Correct 50, Time 0.10848736763000488 seconds
Epoch 370, Loss 0.5802689924642933, Correct 50, Time 0.10782885551452637 seconds
Epoch 380, Loss 0.11957289169525157, Correct 48, Time 0.11038565635681152 seconds
Epoch 390, Loss 0.9487560787522403, Correct 48, Time 0.10862469673156738 seconds
Epoch 400, Loss 0.3590540472891047, Correct 49, Time 0.10872650146484375 seconds
Epoch 410, Loss 0.8938451136443433, Correct 49, Time 0.11968374252319336 seconds
Epoch 420, Loss 0.5484482082961325, Correct 49, Time 0.11132383346557617 seconds
Epoch 430, Loss 0.9175333953610019, Correct 48, Time 0.21046829223632812 seconds
Epoch 440, Loss 0.6102923969649474, Correct 48, Time 0.11140823364257812 seconds
Epoch 450, Loss 0.30821237295103393, Correct 48, Time 0.10830974578857422 seconds
Epoch 460, Loss 0.21374991640025437, Correct 48, Time 0.10860872268676758 seconds
Epoch 470, Loss 0.530832280197137, Correct 50, Time 0.10904359817504883 seconds
Epoch 480, Loss 0.27808911924563084, Correct 49, Time 0.10838723182678223 seconds
Epoch 490, Loss 0.3072473114850978, Correct 50, Time 0.10948848724365234 seconds
Average Time Per Epoch: 0.12288490182651068 s
```

Xor:
```
Epoch 0, Loss 6.330395483717392, Correct 32, Time 14.255102634429932 seconds
Epoch 10, Loss 4.522926814369849, Correct 38, Time 0.11731362342834473 seconds
Epoch 20, Loss 6.033851109850486, Correct 39, Time 0.10964202880859375 seconds
Epoch 30, Loss 5.2981015374684635, Correct 39, Time 0.10764813423156738 seconds
Epoch 40, Loss 3.565030783238192, Correct 42, Time 0.12564659118652344 seconds
Epoch 50, Loss 4.322137877115235, Correct 43, Time 0.10838651657104492 seconds
Epoch 60, Loss 3.144660140998094, Correct 44, Time 0.10946393013000488 seconds
Epoch 70, Loss 1.9136282591030316, Correct 44, Time 0.11581993103027344 seconds
Epoch 80, Loss 3.790563776897265, Correct 44, Time 0.10708498954772949 seconds
Epoch 90, Loss 3.4669194129731, Correct 45, Time 0.18593525886535645 seconds
Epoch 100, Loss 3.3795914357422565, Correct 45, Time 0.10857987403869629 seconds
Epoch 110, Loss 2.618916304006704, Correct 46, Time 0.11489033699035645 seconds
Epoch 120, Loss 3.760744888943395, Correct 46, Time 0.11970353126525879 seconds
Epoch 130, Loss 1.2841682855330592, Correct 47, Time 0.11089920997619629 seconds
Epoch 140, Loss 1.2517501950305414, Correct 48, Time 0.11299538612365723 seconds
Epoch 150, Loss 0.5061318962278344, Correct 50, Time 0.11050271987915039 seconds
Epoch 160, Loss 2.2243924033906857, Correct 50, Time 0.10675859451293945 seconds
Epoch 170, Loss 3.0457855680580685, Correct 47, Time 0.10787796974182129 seconds
Epoch 180, Loss 0.906038542095847, Correct 44, Time 0.10690641403198242 seconds
Epoch 190, Loss 1.1885709392256936, Correct 48, Time 0.21114873886108398 seconds
Epoch 200, Loss 1.2998819690547738, Correct 50, Time 0.10946226119995117 seconds
Epoch 210, Loss 3.092138526417445, Correct 48, Time 0.11265730857849121 seconds
Epoch 220, Loss 1.5107374091760575, Correct 49, Time 0.10872840881347656 seconds
Epoch 230, Loss 0.6783202842142986, Correct 49, Time 0.12560749053955078 seconds
Epoch 240, Loss 0.632316026905082, Correct 49, Time 0.10735869407653809 seconds
Epoch 250, Loss 0.8958843461775204, Correct 49, Time 0.1103665828704834 seconds
Epoch 260, Loss 0.37855194556974586, Correct 50, Time 0.11151528358459473 seconds
Epoch 270, Loss 0.8681709402072038, Correct 49, Time 0.1096656322479248 seconds
Epoch 280, Loss 1.1860059873742363, Correct 49, Time 0.10878849029541016 seconds
Epoch 290, Loss 0.4675737495974307, Correct 49, Time 0.23073291778564453 seconds
Epoch 300, Loss 0.5245558919021283, Correct 49, Time 0.10845422744750977 seconds
Epoch 310, Loss 0.2949532747366753, Correct 50, Time 0.1098785400390625 seconds
Epoch 320, Loss 1.328710515468395, Correct 49, Time 0.1100459098815918 seconds
Epoch 330, Loss 0.16886668064703475, Correct 50, Time 0.1082906723022461 seconds
Epoch 340, Loss 1.7359817233793504, Correct 48, Time 0.10859894752502441 seconds
Epoch 350, Loss 0.3135123014403213, Correct 49, Time 0.10877752304077148 seconds
Epoch 360, Loss 0.3543110389006435, Correct 49, Time 0.11185193061828613 seconds
Epoch 370, Loss 0.08025021144997783, Correct 50, Time 0.10779428482055664 seconds
Epoch 380, Loss 1.282671063499808, Correct 49, Time 0.10973668098449707 seconds
Epoch 390, Loss 0.4511280129746982, Correct 50, Time 0.2001054286956787 seconds
Epoch 400, Loss 0.8388394567501151, Correct 50, Time 0.1091458797454834 seconds
Epoch 410, Loss 1.2248139766243045, Correct 50, Time 0.10737156867980957 seconds
Epoch 420, Loss 0.10248782533802753, Correct 50, Time 0.1120138168334961 seconds
Epoch 430, Loss 0.8405103566014585, Correct 50, Time 0.11140108108520508 seconds
Epoch 440, Loss 0.0520065555431333, Correct 49, Time 0.1092064380645752 seconds
Epoch 450, Loss 0.2788104831396997, Correct 50, Time 0.1091153621673584 seconds
Epoch 460, Loss 1.1094304894596505, Correct 49, Time 0.11794233322143555 seconds
Epoch 470, Loss 1.0666833106130607, Correct 50, Time 0.10792422294616699 seconds
Epoch 480, Loss 0.1153103444881602, Correct 49, Time 0.10874223709106445 seconds
Epoch 490, Loss 1.394527870999093, Correct 49, Time 0.25347256660461426 seconds
Average Time Per Epoch: 0.12415481187059789 s
```

## GPU

50 Points, 100 Hidden Layers, 0.05 Learning Rate

Simple:
```
Epoch 0, Loss 5.278902827194052, Correct 40, Time 3.745617151260376 seconds
Epoch 10, Loss 2.3586122117325985, Correct 48, Time 1.491678237915039 seconds
Epoch 20, Loss 1.9123776504176015, Correct 50, Time 1.4781379699707031 seconds
Epoch 30, Loss 1.001524837750129, Correct 50, Time 1.4909920692443848 seconds
Epoch 40, Loss 1.897019363970908, Correct 48, Time 1.952913522720337 seconds
Epoch 50, Loss 1.3037767971572367, Correct 50, Time 1.5029211044311523 seconds
Epoch 60, Loss 0.5233599366303412, Correct 49, Time 1.4830543994903564 seconds
Epoch 70, Loss 0.4629375711815814, Correct 49, Time 2.081329345703125 seconds
Epoch 80, Loss 0.3746387436983035, Correct 50, Time 1.5443611145019531 seconds
Epoch 90, Loss 0.49332404764694443, Correct 50, Time 1.4832556247711182 seconds
Epoch 100, Loss 1.4207515042749759, Correct 50, Time 1.6610801219940186 seconds
Epoch 110, Loss 0.3635243871469989, Correct 50, Time 1.4737401008605957 seconds
Epoch 120, Loss 0.41784099132492913, Correct 50, Time 1.5314011573791504 seconds
Epoch 130, Loss 0.933609568629566, Correct 50, Time 1.5308449268341064 seconds
Epoch 140, Loss 0.22355869840211456, Correct 50, Time 1.6254527568817139 seconds
Epoch 150, Loss 0.08679728348782417, Correct 50, Time 1.4745047092437744 seconds
Epoch 160, Loss 0.9262107987368972, Correct 50, Time 1.5372095108032227 seconds
Epoch 170, Loss 0.003638978342173486, Correct 50, Time 2.1224966049194336 seconds
Epoch 180, Loss 0.023391973765925535, Correct 50, Time 1.4789092540740967 seconds
Epoch 190, Loss 0.260916274303701, Correct 50, Time 1.4688694477081299 seconds
Epoch 200, Loss 0.037435862862602635, Correct 50, Time 1.818638801574707 seconds
Epoch 210, Loss 0.0888944534850349, Correct 50, Time 1.499218463897705 seconds
Epoch 220, Loss 0.15304586707876208, Correct 50, Time 1.5103347301483154 seconds
Epoch 230, Loss 0.13779762383160757, Correct 50, Time 1.4747390747070312 seconds
Epoch 240, Loss 0.5383963438988126, Correct 50, Time 1.4699218273162842 seconds
Epoch 250, Loss 0.05578272600800325, Correct 50, Time 1.5652039051055908 seconds
Epoch 260, Loss 0.2127971442994836, Correct 50, Time 1.4661765098571777 seconds
Epoch 270, Loss 0.11092804298075011, Correct 50, Time 1.8077645301818848 seconds
Epoch 280, Loss 0.10803297506618739, Correct 50, Time 1.5201942920684814 seconds
Epoch 290, Loss 0.5183938126356166, Correct 50, Time 1.5516011714935303 seconds
Epoch 300, Loss 0.586080242659188, Correct 50, Time 2.173379421234131 seconds
Epoch 310, Loss 0.5522209507112735, Correct 50, Time 1.4728655815124512 seconds
Epoch 320, Loss 0.016328321901019235, Correct 50, Time 1.4729466438293457 seconds
Epoch 330, Loss 0.5950938699850878, Correct 50, Time 1.9365370273590088 seconds
Epoch 340, Loss 0.029861351347236366, Correct 50, Time 1.4836812019348145 seconds
Epoch 350, Loss 0.004467418633888215, Correct 50, Time 1.4926624298095703 seconds
Epoch 360, Loss 0.4120263159303525, Correct 50, Time 1.546553611755371 seconds
Epoch 370, Loss 0.1842991237514763, Correct 50, Time 1.577977180480957 seconds
Epoch 380, Loss 0.014379975128427881, Correct 50, Time 1.493624210357666 seconds
Epoch 390, Loss 0.1094541756691865, Correct 50, Time 1.504605770111084 seconds
Epoch 400, Loss 0.0070407574501047746, Correct 50, Time 1.6527974605560303 seconds
Epoch 410, Loss 0.3791660285200534, Correct 50, Time 1.5766594409942627 seconds
Epoch 420, Loss 0.512137311424785, Correct 50, Time 1.4981110095977783 seconds
Epoch 430, Loss 0.4254838237457636, Correct 50, Time 1.9711840152740479 seconds
Epoch 440, Loss 0.10821795391880991, Correct 50, Time 1.494149923324585 seconds
Epoch 450, Loss 0.16972596359864087, Correct 50, Time 1.5436570644378662 seconds
Epoch 460, Loss 0.40524467951392973, Correct 50, Time 2.1411821842193604 seconds
Epoch 470, Loss 0.011276121882437113, Correct 50, Time 1.5003538131713867 seconds
Epoch 480, Loss 0.036039447640099566, Correct 50, Time 1.4667108058929443 seconds
Epoch 490, Loss 0.005542253672783818, Correct 50, Time 1.8449413776397705 seconds
Average Time Per Epoch: 1.6108180721680483 s
```

Split:
```
Epoch 0, Loss 7.953730385076907, Correct 30, Time 3.820286750793457 seconds
Epoch 10, Loss 9.322674101352403, Correct 38, Time 1.6754348278045654 seconds
Epoch 20, Loss 8.055659462210391, Correct 28, Time 1.5007328987121582 seconds
Epoch 30, Loss 4.918962306645864, Correct 43, Time 1.48915696144104 seconds
Epoch 40, Loss 3.9077804059967796, Correct 44, Time 1.5848197937011719 seconds
Epoch 50, Loss 2.9267700975925415, Correct 48, Time 1.5193252563476562 seconds
Epoch 60, Loss 3.776593200942429, Correct 49, Time 1.4955284595489502 seconds
Epoch 70, Loss 2.6432513208801325, Correct 48, Time 1.681694746017456 seconds
Epoch 80, Loss 2.191945461425086, Correct 49, Time 1.6351745128631592 seconds
Epoch 90, Loss 2.1618379161313506, Correct 48, Time 1.517437219619751 seconds
Epoch 100, Loss 2.524099072464156, Correct 48, Time 1.7222552299499512 seconds
Epoch 110, Loss 1.5620815398086467, Correct 49, Time 1.4956514835357666 seconds
Epoch 120, Loss 1.3279446364696486, Correct 49, Time 1.5946052074432373 seconds
Epoch 130, Loss 1.3922255482734174, Correct 50, Time 1.4789116382598877 seconds
Epoch 140, Loss 2.2789010577983526, Correct 50, Time 1.520258903503418 seconds
Epoch 150, Loss 0.4793474800183593, Correct 50, Time 1.4822745323181152 seconds
Epoch 160, Loss 1.0311034163682435, Correct 50, Time 1.6412570476531982 seconds
Epoch 170, Loss 1.660562695104482, Correct 49, Time 1.8196215629577637 seconds
Epoch 180, Loss 1.4483136365133262, Correct 48, Time 1.477949857711792 seconds
Epoch 190, Loss 0.2944459222940665, Correct 50, Time 1.4899065494537354 seconds
Epoch 200, Loss 0.8408650138351882, Correct 50, Time 2.2352633476257324 seconds
Epoch 210, Loss 0.7885459233598628, Correct 50, Time 1.482713222503662 seconds
Epoch 220, Loss 0.22454400747437359, Correct 50, Time 1.5028924942016602 seconds
Epoch 230, Loss 0.22654825950251445, Correct 50, Time 1.8821778297424316 seconds
Epoch 240, Loss 1.2444382791908053, Correct 50, Time 1.5056371688842773 seconds
Epoch 250, Loss 1.1019581593114758, Correct 49, Time 1.5582995414733887 seconds
Epoch 260, Loss 0.1036099793292978, Correct 50, Time 1.5984632968902588 seconds
Epoch 270, Loss 0.5380181019776867, Correct 50, Time 1.5042173862457275 seconds
Epoch 280, Loss 0.31483173123943164, Correct 50, Time 1.4811642169952393 seconds
Epoch 290, Loss 0.964962302593527, Correct 50, Time 1.5943689346313477 seconds
Epoch 300, Loss 0.30914413889023373, Correct 50, Time 1.6544625759124756 seconds
Epoch 310, Loss 0.2265148718553273, Correct 50, Time 1.4851617813110352 seconds
Epoch 320, Loss 0.634979917917667, Correct 50, Time 1.4999220371246338 seconds
Epoch 330, Loss 0.06264338079933687, Correct 50, Time 2.166799783706665 seconds
Epoch 340, Loss 0.08127709940440173, Correct 50, Time 1.480811595916748 seconds
Epoch 350, Loss 0.45561905232858935, Correct 50, Time 1.5009143352508545 seconds
Epoch 360, Loss 0.2734405481736473, Correct 50, Time 1.924868106842041 seconds
Epoch 370, Loss 0.592610010875559, Correct 50, Time 1.5413990020751953 seconds
Epoch 380, Loss 0.29432809591029474, Correct 50, Time 1.515387773513794 seconds
Epoch 390, Loss 0.4293751925808454, Correct 50, Time 1.46870756149292 seconds
Epoch 400, Loss 0.16848611573338704, Correct 50, Time 1.4668562412261963 seconds
Epoch 410, Loss 0.8750858573283726, Correct 50, Time 1.5193743705749512 seconds
Epoch 420, Loss 0.1693939142340325, Correct 50, Time 1.4750549793243408 seconds
Epoch 430, Loss 0.2142160076003861, Correct 50, Time 2.0895721912384033 seconds
Epoch 440, Loss 0.30665662213744865, Correct 50, Time 1.521958351135254 seconds
Epoch 450, Loss 0.4200736022480158, Correct 50, Time 1.5152392387390137 seconds
Epoch 460, Loss 0.3715020726559454, Correct 50, Time 1.6787302494049072 seconds
Epoch 470, Loss 0.2775764674795482, Correct 50, Time 1.4620308876037598 seconds
Epoch 480, Loss 0.4492500812565175, Correct 50, Time 1.4797248840332031 seconds
Epoch 490, Loss 0.4061232142978837, Correct 50, Time 1.5300571918487549 seconds
Average Time Per Epoch: 1.6163183586869785 s
```

Xor:
```
Epoch 0, Loss 5.9551473064515665, Correct 30, Time 3.7330482006073 seconds
Epoch 10, Loss 5.088088538457905, Correct 40, Time 1.8917419910430908 seconds
Epoch 20, Loss 4.381545160655237, Correct 47, Time 1.478304147720337 seconds
Epoch 30, Loss 2.1850009057445785, Correct 47, Time 1.4753570556640625 seconds
Epoch 40, Loss 2.1434730240850786, Correct 46, Time 1.8960444927215576 seconds
Epoch 50, Loss 1.8039963433654067, Correct 47, Time 1.4648151397705078 seconds
Epoch 60, Loss 1.9256875861912135, Correct 48, Time 1.4558234214782715 seconds
Epoch 70, Loss 1.4444138335408319, Correct 47, Time 1.4592134952545166 seconds
Epoch 80, Loss 1.3915407357109952, Correct 48, Time 1.7192084789276123 seconds
Epoch 90, Loss 0.6354018091061518, Correct 48, Time 1.4673023223876953 seconds
Epoch 100, Loss 1.9553657077094904, Correct 48, Time 1.4856276512145996 seconds
Epoch 110, Loss 1.7960544450861664, Correct 48, Time 2.197545289993286 seconds
Epoch 120, Loss 2.5875219081048932, Correct 49, Time 1.6379213333129883 seconds
Epoch 130, Loss 0.42048885802968566, Correct 48, Time 1.4581024646759033 seconds
Epoch 140, Loss 0.32235163923087856, Correct 48, Time 1.5322723388671875 seconds
Epoch 150, Loss 1.6902979863623224, Correct 49, Time 1.4682223796844482 seconds
Epoch 160, Loss 1.4069958193275647, Correct 48, Time 1.5287833213806152 seconds
Epoch 170, Loss 1.2725647518582068, Correct 49, Time 1.4562866687774658 seconds
Epoch 180, Loss 0.6755083459935554, Correct 50, Time 2.0278186798095703 seconds
Epoch 190, Loss 1.7945956248514774, Correct 49, Time 1.475583553314209 seconds
Epoch 200, Loss 1.937129430453209, Correct 49, Time 1.5264620780944824 seconds
Epoch 210, Loss 1.0210093758445102, Correct 49, Time 1.6324138641357422 seconds
Epoch 220, Loss 0.6153068183937475, Correct 49, Time 1.4617924690246582 seconds
Epoch 230, Loss 0.4441410170508846, Correct 49, Time 1.4749736785888672 seconds
Epoch 240, Loss 0.1609777612559275, Correct 49, Time 1.463444471359253 seconds
Epoch 250, Loss 1.4818078996361672, Correct 49, Time 2.0299670696258545 seconds
Epoch 260, Loss 0.03754493135076891, Correct 49, Time 1.457435131072998 seconds
Epoch 270, Loss 0.3452848795146832, Correct 49, Time 1.5161986351013184 seconds
Epoch 280, Loss 0.672601969383849, Correct 49, Time 1.7535626888275146 seconds
Epoch 290, Loss 0.6964994290253519, Correct 49, Time 1.560314655303955 seconds
Epoch 300, Loss 1.2107124221797942, Correct 49, Time 1.4659502506256104 seconds
Epoch 310, Loss 1.1417302199950676, Correct 49, Time 1.4719879627227783 seconds
Epoch 320, Loss 0.24255293803915992, Correct 50, Time 1.8564958572387695 seconds
Epoch 330, Loss 1.2359720094018094, Correct 48, Time 1.535376787185669 seconds
Epoch 340, Loss 0.7654122552336882, Correct 50, Time 1.457158088684082 seconds
Epoch 350, Loss 1.5808895801618805, Correct 50, Time 1.7853007316589355 seconds
Epoch 360, Loss 0.15564475055879357, Correct 49, Time 1.5056648254394531 seconds
Epoch 370, Loss 0.35791547172780985, Correct 49, Time 1.5622341632843018 seconds
Epoch 380, Loss 0.9082748914934071, Correct 50, Time 1.4964568614959717 seconds
Epoch 390, Loss 0.07817554598596166, Correct 49, Time 1.5011487007141113 seconds
Epoch 400, Loss 1.5531713654299042, Correct 49, Time 1.507669448852539 seconds
Epoch 410, Loss 0.19913814945639527, Correct 50, Time 1.5478055477142334 seconds
Epoch 420, Loss 1.7434340268463988, Correct 50, Time 1.9300041198730469 seconds
Epoch 430, Loss 0.15399957694733127, Correct 49, Time 1.4768726825714111 seconds
Epoch 440, Loss 1.414568371850062, Correct 49, Time 1.4794378280639648 seconds
Epoch 450, Loss 0.45119899241237293, Correct 50, Time 1.7665235996246338 seconds
Epoch 460, Loss 0.7257686862909527, Correct 50, Time 1.4760267734527588 seconds
Epoch 470, Loss 1.3331666868108658, Correct 50, Time 1.4547836780548096 seconds
Epoch 480, Loss 0.18763078175157116, Correct 50, Time 1.4690799713134766 seconds
Epoch 490, Loss 1.770760320455593, Correct 49, Time 1.9482669830322266 seconds
Average Time Per Epoch: 1.5894891518151355 s
```

# Bigger Model

50 Points, 200 Hidden Layers, 0.05 Learning Rate, Split Dataset

CPU:
```
Epoch 0, Loss 9.872976339070592, Correct 33, Time 15.022867918014526 seconds
Epoch 10, Loss 3.782152479850833, Correct 38, Time 0.2507197856903076 seconds
Epoch 20, Loss 2.5817567027400403, Correct 44, Time 0.2491755485534668 seconds
Epoch 30, Loss 3.3175461775452337, Correct 42, Time 0.26083898544311523 seconds
Epoch 40, Loss 2.399787955516848, Correct 46, Time 0.2534451484680176 seconds
Epoch 50, Loss 1.9799690552182947, Correct 49, Time 0.2620811462402344 seconds
Epoch 60, Loss 3.629292512108554, Correct 49, Time 0.2545492649078369 seconds
Epoch 70, Loss 0.8705102236072713, Correct 49, Time 0.44315481185913086 seconds
Epoch 80, Loss 1.4656785731803046, Correct 49, Time 0.25194883346557617 seconds
Epoch 90, Loss 1.2958637858657955, Correct 49, Time 0.2760434150695801 seconds
Epoch 100, Loss 0.9597352420964207, Correct 49, Time 0.2527174949645996 seconds
Epoch 110, Loss 1.0649515533394331, Correct 50, Time 0.2926051616668701 seconds
Epoch 120, Loss 1.614176782806922, Correct 50, Time 0.2528531551361084 seconds
Epoch 130, Loss 1.7870142608641317, Correct 49, Time 0.24842071533203125 seconds
Epoch 140, Loss 0.13095987128759198, Correct 49, Time 0.25554966926574707 seconds
Epoch 150, Loss 0.723271263499143, Correct 48, Time 0.24854779243469238 seconds
Epoch 160, Loss 1.2372848295700265, Correct 49, Time 0.2616443634033203 seconds
Epoch 170, Loss 0.9387156603044126, Correct 49, Time 0.25080394744873047 seconds
Epoch 180, Loss 0.7847990779662375, Correct 49, Time 0.2817349433898926 seconds
Epoch 190, Loss 2.0872638409229367, Correct 48, Time 0.25065016746520996 seconds
Epoch 200, Loss 0.4869644677105778, Correct 49, Time 0.5098791122436523 seconds
Epoch 210, Loss 0.5300064466810547, Correct 50, Time 0.24832725524902344 seconds
Epoch 220, Loss 2.78365155479379, Correct 50, Time 0.2583956718444824 seconds
Epoch 230, Loss 0.5234747975612898, Correct 50, Time 0.24741530418395996 seconds
Epoch 240, Loss 1.1032363303240367, Correct 50, Time 0.25417494773864746 seconds
Epoch 250, Loss 0.18407135107597317, Correct 49, Time 0.24943089485168457 seconds
Epoch 260, Loss 1.1236640543631566, Correct 50, Time 0.26171875 seconds
Epoch 270, Loss 0.99386458758955, Correct 50, Time 0.24941420555114746 seconds
Epoch 280, Loss 1.0554231039638053, Correct 50, Time 0.2609701156616211 seconds
Epoch 290, Loss 0.6857722350534481, Correct 50, Time 0.5293025970458984 seconds
Epoch 300, Loss 0.41749403456519363, Correct 50, Time 0.2606184482574463 seconds
Epoch 310, Loss 0.8166164911892846, Correct 50, Time 0.2491140365600586 seconds
Epoch 320, Loss 0.3984937007160288, Correct 50, Time 0.2607114315032959 seconds
Epoch 330, Loss 0.20970548358227337, Correct 50, Time 0.4049868583679199 seconds
Epoch 340, Loss 0.5532838757047928, Correct 50, Time 0.25902533531188965 seconds
Epoch 350, Loss 0.07785243051869864, Correct 50, Time 0.2513458728790283 seconds
Epoch 360, Loss 0.48108765543415977, Correct 50, Time 0.2714095115661621 seconds
Epoch 370, Loss 0.8165444831702153, Correct 50, Time 0.24884033203125 seconds
Epoch 380, Loss 0.3013734066177798, Correct 50, Time 0.255126953125 seconds
Epoch 390, Loss 0.31548342262128004, Correct 50, Time 0.25196146965026855 seconds
Epoch 400, Loss 0.37629869782814745, Correct 50, Time 0.25966882705688477 seconds
Epoch 410, Loss 0.7151182921967829, Correct 50, Time 0.25223207473754883 seconds
Epoch 420, Loss 0.23941095869228812, Correct 50, Time 0.5182526111602783 seconds
Epoch 430, Loss 0.021107019289059743, Correct 50, Time 0.2471332550048828 seconds
Epoch 440, Loss 0.27321681920996577, Correct 50, Time 0.2505829334259033 seconds
Epoch 450, Loss 0.3878368704216091, Correct 50, Time 0.24921298027038574 seconds
Epoch 460, Loss 0.46496372008570863, Correct 50, Time 0.2617332935333252 seconds
Epoch 470, Loss 0.1968130745270954, Correct 50, Time 0.24824786186218262 seconds
Epoch 480, Loss 0.21150812114923262, Correct 50, Time 0.24964022636413574 seconds
Epoch 490, Loss 1.0355466626627563, Correct 50, Time 0.24869203567504883 seconds
Average Time Per Epoch: 0.28032671951339816 s
```

GPU:
```
Epoch 0, Loss 8.514489900480587, Correct 26, Time 3.553887367248535 seconds
Epoch 10, Loss 3.2343041352992863, Correct 47, Time 2.1752655506134033 seconds
Epoch 20, Loss 2.181412544426332, Correct 49, Time 1.5624980926513672 seconds
Epoch 30, Loss 2.560387065780779, Correct 48, Time 1.5595989227294922 seconds
Epoch 40, Loss 1.1751339903711158, Correct 49, Time 1.9696786403656006 seconds
Epoch 50, Loss 1.2622374906702558, Correct 50, Time 1.5657696723937988 seconds
Epoch 60, Loss 1.102139558212079, Correct 50, Time 1.6275255680084229 seconds
Epoch 70, Loss 0.9885666286461086, Correct 48, Time 1.554772138595581 seconds
Epoch 80, Loss 1.494709573204365, Correct 50, Time 1.6221766471862793 seconds
Epoch 90, Loss 1.4043190636628098, Correct 50, Time 1.9874544143676758 seconds
Epoch 100, Loss 0.5332608019518037, Correct 50, Time 1.558100700378418 seconds
Epoch 110, Loss 0.5147217599931497, Correct 50, Time 1.5538218021392822 seconds
Epoch 120, Loss 0.6448987554275116, Correct 50, Time 2.080397367477417 seconds
Epoch 130, Loss 1.1754303988268728, Correct 50, Time 1.5467033386230469 seconds
Epoch 140, Loss 0.6716025799900446, Correct 50, Time 1.561486005783081 seconds
Epoch 150, Loss 0.13371952757991712, Correct 48, Time 1.5476362705230713 seconds
Epoch 160, Loss 1.056920715988037, Correct 49, Time 1.6142981052398682 seconds
Epoch 170, Loss 1.4507001178781975, Correct 50, Time 1.8252193927764893 seconds
Epoch 180, Loss 0.18652631933427172, Correct 50, Time 1.5329155921936035 seconds
Epoch 190, Loss 0.13969003682275355, Correct 50, Time 1.5509717464447021 seconds
Epoch 200, Loss 0.8080852454961825, Correct 50, Time 2.433527708053589 seconds
Epoch 210, Loss 0.12573686642011758, Correct 50, Time 1.571418046951294 seconds
Epoch 220, Loss 0.6790569627503944, Correct 50, Time 1.599733591079712 seconds
Epoch 230, Loss 0.06689048052308344, Correct 50, Time 1.5777592658996582 seconds
Epoch 240, Loss 0.2294882524085983, Correct 50, Time 1.5793395042419434 seconds
Epoch 250, Loss 0.3958272936448399, Correct 50, Time 2.11755633354187 seconds
Epoch 260, Loss 0.329290837313369, Correct 50, Time 1.6015663146972656 seconds
Epoch 270, Loss 0.4993293463854042, Correct 50, Time 1.5955626964569092 seconds
Epoch 280, Loss 0.21847676050891657, Correct 50, Time 1.7802801132202148 seconds
Epoch 290, Loss 0.30761260339271795, Correct 50, Time 1.690258264541626 seconds
Epoch 300, Loss 0.11947701378878069, Correct 50, Time 1.6610050201416016 seconds
Epoch 310, Loss 0.3022083567363251, Correct 50, Time 1.5604522228240967 seconds
Epoch 320, Loss 0.3032654415537201, Correct 50, Time 1.5695490837097168 seconds
Epoch 330, Loss 0.24177911252670162, Correct 50, Time 2.3166518211364746 seconds
Epoch 340, Loss 0.14040309031398276, Correct 50, Time 1.6009089946746826 seconds
Epoch 350, Loss 0.27301117389750357, Correct 50, Time 1.5699901580810547 seconds
Epoch 360, Loss 0.16502062490375902, Correct 50, Time 1.5687346458435059 seconds
Epoch 370, Loss 0.061487118414913916, Correct 50, Time 1.6515896320343018 seconds
Epoch 380, Loss 0.13521567994406253, Correct 50, Time 2.119598627090454 seconds
Epoch 390, Loss 0.1855653929079112, Correct 50, Time 1.563878059387207 seconds
Epoch 400, Loss 0.2838562686060855, Correct 50, Time 1.5703353881835938 seconds
Epoch 410, Loss 0.23626016285734872, Correct 50, Time 1.850867509841919 seconds
Epoch 420, Loss 0.25226210246472947, Correct 50, Time 1.5895214080810547 seconds
Epoch 430, Loss 0.001701472633548726, Correct 50, Time 1.7596755027770996 seconds
Epoch 440, Loss 0.12808964911628834, Correct 50, Time 1.5693397521972656 seconds
Epoch 450, Loss 0.53272374100084, Correct 50, Time 1.6375620365142822 seconds
Epoch 460, Loss 0.34853443780876603, Correct 50, Time 2.173124313354492 seconds
Epoch 470, Loss 0.01875506119622752, Correct 50, Time 1.5824456214904785 seconds
Epoch 480, Loss 0.25803236296543747, Correct 50, Time 1.6110270023345947 seconds
Epoch 490, Loss 0.03645844108493376, Correct 50, Time 1.6488242149353027 seconds
Average Time Per Epoch: 1.7021409270758618 s
```