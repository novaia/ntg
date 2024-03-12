{
    description = "Generative neural networks for 3D terrain";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
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
            pyVer = "311";
            py = "python${pyVer}";
            overlays = [
                (final: prev: {
                    ${py} = prev.${py}.override {
                        packageOverrides = final2: prev2: {
                            jaxlib = prev2.jaxlib.overrideAttrs (oldAttrs: {
                                #fetchAttrs = oldAttrs.fetchAttrs // {
                                #    sha256.x86_64-linux = "sha256-h4zE+Z6z7odg7Avr54pgsjInBaHf+BqVQUi4SsV3Nqo=";
                                #};
                                sha256.x86_64-linux = "sha256-h4zE+Z6z7odg7Avr54pgsjInBaHf+BqVQUi4SsV3Nqo=";
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
                ];
            };
        };
    });
}
