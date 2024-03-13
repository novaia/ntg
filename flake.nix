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
                                # Replace jaxlib' with jaxlib-bin in nativeCheckInputs so that jaxlib is never used.
                                #nativeCheckInputs = with prevPkgs; [
                                #    jaxlib-bin
                                #    matplotlib
                                #    pytestCheckHook
                                #    pytest-xdist
                                #];
                                nativeCheckInputs = [];
                                pythonImportsCheck = [];
                                pytestFlagsArray = [];
                                passthru.tests = [];
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
