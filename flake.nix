{
    description = "Generative neural networks for 3D terrain";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/23.11";
        nixpkgs-unstable.url = "github:nixos/nixpkgs/nixpkgs-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };
    outputs = inputs@{ self, nixpkgs, nixpkgs-unstable, flake-utils, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system: {
        devShells = let
            pyVer = "311";
            py = "python${pyVer}";
            overlays = [
                (final: prev: {
                    ${py} = prev.${py}.override {
                        packageOverrides = finalPkgs: prevPkgs: {
                            jax = prevPkgs.jax.overridePythonAttrs (o: {
                                nativeCheckInputs = [];
                                pythonImportsCheck = [];
                                pytestFlagsArray = [];
                                passthru.tests = [];
                                doCheck = false;
                            });
                            # For some reason Flax has jaxlib as a builtInput and tensorflow as a nativeCheckInput,
                            # so set jaxlib to jaxlib-bin in order to avoid building jaxlib and turn off all checks
                            # to avoid building tensorflow.
                            jaxlib = prevPkgs.jaxlib-bin;
                            flax = prevPkgs.flax.overridePythonAttrs (o: {
                                nativeCheckInputs = [];
                                pythonImportsCheck = [];
                                pytestFlagsArray = [];
                                doCheck = false;
                            });
                            wandb = prevPkgs.wandb.overridePythonAttrs(o: {
                                nativeCheckInputs = [];
                                pythonIMportsCheck = [];
                                doCheck = false;
                            });
                        };
                    };
                })
            ];
            unstableCudaPkgs = import nixpkgs-unstable {
                inherit system overlays;
                config = {
                    allowUnfree = true;
                    cudaSupport = true;
                };
            };
        in rec {
            default = unstableCudaPkgs.mkShell {
                name = "cuda";
                buildInputs = [
                    (unstableCudaPkgs.${py}.withPackages (pyp: with pyp; [
                        jax
                        jaxlib-bin
                        flax
                        pyarrow
                        pillow
                        pandas
                        datasets
                        wandb
                    ]))
                    unstableCudaPkgs.cudaPackages.cudatoolkit
                    unstableCudaPkgs.cudaPackages.cuda_cudart
                    unstableCudaPkgs.cudaPackages.cudnn
                    unstableCudaPkgs.linuxPackages.nvidia_x11
                ];
                shellHook = ''
                    export CUDA_PATH=${unstableCudaPkgs.cudatoolkit}
                    export EXTRA_LDFLAGS="-L/lib -L${unstableCudaPkgs.linuxPackages.nvidia_x11}/lib"
                    export LD_LIBRARY_PATH=/run/opengl-driver/lib/
                '';
            };
        };
    });
}
