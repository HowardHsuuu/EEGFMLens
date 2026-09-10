# Internal correction experiment

All effects below are paired equal-subject balanced-accuracy changes relative to the original fixed head on the same inputs. Maps were fitted only on five training subjects at gains 0.5 and 2, selected on two validation subjects, and evaluated on the held-out subject. The reported gains 0.75 and 1.5 were not used for fitting or selection. Gain 1 measures clean-condition harm. No learned correction uses a clean test donor.

The table reports every site and method; intervals are descriptive 95% subject bootstraps with only eight subjects, not corrected for multiple testing. Identity, oracle and full-final-site recovery checks passed during native execution. Oracle results are stored separately in the machine-readable file; restoring clean activations is not itself a learned repair.

| Model / site / correction | Gain 0.75 | Clean effect | Gain 1.5 |
|---|---:|---:|---:|

| cbramod / embedding.output / mean | -0.119 [-0.192, -0.052] | -0.079 [-0.124, -0.021] | +0.048 [-0.002, +0.108] |

| cbramod / embedding.output / random_mean | -0.149 [-0.279, -0.045] | -0.154 [-0.239, -0.093] | -0.053 [-0.116, +0.017] |

| cbramod / embedding.output / affine | -0.020 [-0.113, +0.055] | -0.076 [-0.135, -0.023] | +0.026 [-0.024, +0.077] |

| cbramod / embedding.output / permuted_affine | -0.063 [-0.136, +0.000] | -0.076 [-0.129, -0.021] | -0.049 [-0.114, +0.022] |

| cbramod / blocks.2.output / mean | -0.027 [-0.094, +0.029] | -0.051 [-0.084, -0.020] | +0.021 [+0.001, +0.043] |

| cbramod / blocks.2.output / random_mean | -0.051 [-0.137, +0.012] | -0.082 [-0.141, -0.038] | -0.040 [-0.091, +0.011] |

| cbramod / blocks.2.output / affine | +0.027 [-0.070, +0.119] | -0.041 [-0.120, +0.022] | +0.031 [-0.017, +0.085] |

| cbramod / blocks.2.output / permuted_affine | -0.051 [-0.137, +0.014] | -0.024 [-0.079, +0.021] | -0.109 [-0.140, -0.079] |

| cbramod / blocks.5.output / mean | +0.017 [-0.010, +0.045] | -0.018 [-0.036, +0.001] | +0.002 [-0.013, +0.017] |

| cbramod / blocks.5.output / random_mean | -0.062 [-0.159, +0.024] | -0.067 [-0.158, +0.013] | -0.014 [-0.051, +0.025] |

| cbramod / blocks.5.output / affine | -0.006 [-0.107, +0.084] | -0.068 [-0.163, +0.014] | +0.017 [-0.031, +0.073] |

| cbramod / blocks.5.output / permuted_affine | -0.044 [-0.122, +0.020] | -0.044 [-0.078, -0.010] | -0.108 [-0.155, -0.063] |

| cbramod / blocks.8.output / mean | +0.006 [-0.028, +0.037] | -0.026 [-0.047, -0.008] | +0.008 [-0.026, +0.042] |

| cbramod / blocks.8.output / random_mean | -0.114 [-0.198, -0.041] | -0.136 [-0.203, -0.069] | -0.058 [-0.124, -0.000] |

| cbramod / blocks.8.output / affine | -0.008 [-0.115, +0.085] | -0.091 [-0.181, -0.021] | -0.019 [-0.071, +0.038] |

| cbramod / blocks.8.output / permuted_affine | -0.055 [-0.153, +0.026] | -0.122 [-0.188, -0.053] | -0.067 [-0.109, -0.025] |

| cbramod / blocks.11.output / mean | +0.020 [-0.047, +0.077] | -0.044 [-0.066, -0.023] | -0.009 [-0.059, +0.043] |

| cbramod / blocks.11.output / random_mean | -0.040 [-0.092, +0.006] | -0.062 [-0.126, +0.004] | -0.051 [-0.095, -0.008] |

| cbramod / blocks.11.output / affine | +0.021 [-0.088, +0.120] | -0.064 [-0.147, +0.008] | +0.014 [-0.026, +0.056] |

| cbramod / blocks.11.output / permuted_affine | +0.026 [+0.007, +0.045] | +0.000 [-0.017, +0.020] | +0.018 [+0.004, +0.029] |

| labram / embedding.output / mean | +0.004 [+0.001, +0.008] | +0.001 [+0.000, +0.003] | +0.003 [+0.000, +0.008] |

| labram / embedding.output / random_mean | +0.005 [+0.001, +0.010] | -0.001 [-0.006, +0.003] | +0.001 [-0.002, +0.005] |

| labram / embedding.output / affine | -0.005 [-0.023, +0.015] | -0.005 [-0.020, +0.011] | -0.002 [-0.011, +0.007] |

| labram / embedding.output / permuted_affine | -0.007 [-0.020, +0.005] | -0.002 [-0.014, +0.012] | -0.002 [-0.008, +0.005] |

| labram / blocks.2.output / mean | +0.005 [-0.011, +0.022] | -0.002 [-0.008, +0.004] | +0.003 [-0.010, +0.023] |

| labram / blocks.2.output / random_mean | -0.006 [-0.016, +0.006] | -0.001 [-0.012, +0.010] | +0.002 [-0.007, +0.011] |

| labram / blocks.2.output / affine | +0.009 [-0.021, +0.040] | +0.006 [-0.013, +0.025] | +0.009 [-0.008, +0.032] |

| labram / blocks.2.output / permuted_affine | +0.007 [-0.034, +0.049] | -0.005 [-0.039, +0.030] | -0.006 [-0.023, +0.010] |

| labram / blocks.5.output / mean | +0.013 [-0.002, +0.030] | -0.001 [-0.010, +0.008] | +0.000 [-0.013, +0.018] |

| labram / blocks.5.output / random_mean | +0.007 [-0.005, +0.023] | +0.007 [-0.005, +0.017] | +0.004 [-0.008, +0.014] |

| labram / blocks.5.output / affine | +0.027 [+0.002, +0.061] | +0.015 [-0.010, +0.053] | +0.016 [-0.018, +0.057] |

| labram / blocks.5.output / permuted_affine | -0.012 [-0.041, +0.014] | -0.002 [-0.042, +0.040] | +0.011 [-0.026, +0.047] |

| labram / blocks.8.output / mean | +0.009 [-0.002, +0.020] | +0.001 [-0.015, +0.014] | +0.007 [-0.004, +0.018] |

| labram / blocks.8.output / random_mean | -0.005 [-0.015, +0.004] | +0.008 [-0.008, +0.022] | +0.007 [-0.006, +0.017] |

| labram / blocks.8.output / affine | +0.006 [-0.013, +0.025] | +0.011 [-0.005, +0.024] | +0.003 [-0.020, +0.021] |

| labram / blocks.8.output / permuted_affine | +0.018 [-0.016, +0.047] | +0.014 [-0.018, +0.049] | +0.015 [-0.011, +0.041] |

| labram / blocks.11.output / mean | +0.009 [-0.008, +0.026] | -0.010 [-0.028, +0.009] | +0.004 [-0.021, +0.035] |

| labram / blocks.11.output / random_mean | -0.002 [-0.020, +0.017] | -0.004 [-0.017, +0.008] | -0.002 [-0.009, +0.006] |

| labram / blocks.11.output / affine | +0.010 [-0.025, +0.038] | -0.004 [-0.026, +0.016] | +0.002 [-0.016, +0.020] |

| labram / blocks.11.output / permuted_affine | +0.002 [-0.022, +0.028] | -0.015 [-0.045, +0.012] | -0.007 [-0.042, +0.035] |
