{
    description = "Generative neural networks for 3D terrain";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/23.11";
        nixpkgs-unstable.url = "github:nixos/nixpkgs/nixpkgs-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };
    outputs = inputs@{ self, nixpkgs, nixpkgs-unstable, flake-utils, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system: let
        inherit (nixpkgs) lib;
        unstableCudaPkgs = import nixpkgs-unstable {
            inherit system;
            config = {
                allowUnfree = true;
                cudaSupport = true;
            };
        };
    in {
        devShells = let
            pyVer = "310";
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
                                doCheck = false;
                            });
                        };
                    };
                })
            ];
            stableJaxPkgs = import nixpkgs {
                inherit system overlays;
                config = {
                    allowUnfree = true;
                    cudaSupport = true;
                };
            };
        in rec {
            default = stableJaxPkgs.mkShell {
                name = "cuda";
                buildInputs = [
                    (stableJaxPkgs.${py}.withPackages (pyp: with pyp; [
                        jax
                        jaxlib-bin
                    ]))
                    unstableCudaPkgs.cudaPackages.cudatoolkit
                ];
                shellHook = ''
                    export LD_LIBRARY_PATH=/run/opengl-driver/lib/
                '';
            };
        };
    });
}
