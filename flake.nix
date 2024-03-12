{
    description = "Generative neural networks for 3D terrain";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/23.11";
        flake-utils.url = "github:numtide/flake-utils";
    };
    outputs = inputs@{ self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system: let
        inherit (nixpkgs) lib;
        basePkgs = import nixpkgs {
            inherit system;
            overlays = [
                self.overlays.default
            ];
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
                                pythonImportsCheck = [];
                                nativeCheckInputs = with prevPkgs; [
                                    #jaxlib'
                                    matplotlib
                                    pytestCheckHook
                                    pytest-xdist
                                ];
                                pytestFlagsArray = [];
                                doCheck = false;
                            });
                        };
                    };
                })
            ];
            cudaPkgs = import nixpkgs {
                inherit system overlays;
                config = {
                    allowUnfree = true;
                    cudaSupport = true;
                };
            };
        in rec {
            default = cudaPkgs.mkShell {
                name = "cuda";
                buildInputs = [
                    (cudaPkgs.${py}.withPackages (pyp: with pyp; [
                        jax
                        jaxlib-bin
                    ]))
                    cudaPkgs.cudatoolkit
                ];
            };
        };
    });
}
