{
  description = "Seminal";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rust-analyzer-src.follows = "";
    };

    flake-utils.url = "github:numtide/flake-utils";

    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, crane, fenix, flake-utils, advisory-db, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let

        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };

        inherit (pkgs) lib;

        toolchain =
          pkgs.rust-bin.stable.latest.default.override
            {
              extensions = [
                "rust-src"
                "rust-analyzer"
                "clippy"
              ];
            };

        craneLib = crane.mkLib pkgs;
        src = craneLib.cleanCargoSource ./.;

        # Common arguments can be set here to avoid repeating them later
        commonArgs = {
          inherit src;
          strictDeps = true;

          buildInputs = [
            # Add additional build inputs here
          ] ++ lib.optionals pkgs.stdenv.isDarwin [
            # Additional darwin specific inputs can be set here
            pkgs.libiconv
          ];

          # Additional environment variables can be set directly
          # MY_CUSTOM_VAR = "some value";
        };

        # Build *just* the cargo dependencies, so we can reuse
        # all of that work (e.g. via cachix) when running in CI
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # Build the actual crate itself, reusing the dependency
        # artifacts from above.
        seminal = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });
      in
      {
        checks = {
          # Build the crate as part of `nix flake check` for convenience
          inherit seminal;

          # Run clippy (and deny all warnings) on the crate source,
          # again, reusing the dependency artifacts from above.
          #
          # Note that this is done as a separate derivation so that
          # we can block the CI if there are issues here, but not
          # prevent downstream consumers from building our crate by itself.
          seminal-clippy = craneLib.cargoClippy (commonArgs // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          });

          seminal-doc = craneLib.cargoDoc (commonArgs // {
            inherit cargoArtifacts;
          });

          # Check formatting
          seminal-fmt = craneLib.cargoFmt {
            inherit src;
          };

          # Audit dependencies
          seminal-audit = craneLib.cargoAudit {
            inherit src advisory-db;
          };

          # Audit licenses
          seminal-deny = craneLib.cargoDeny {
            inherit src;
          };

          # Run tests with cargo-nextest
          # Consider setting `doCheck = false` on `seminal` if you do not want
          # the tests to run twice
          seminal-nextest = craneLib.cargoNextest (commonArgs // {
            inherit cargoArtifacts;
            # partitions = 1;
            # partitionType = "count";
          });
        };

        packages = {
          default = seminal;
        };

        apps.default = flake-utils.lib.mkApp {
          drv = seminal;
        };

        devShells.default = craneLib.devShell {
          # Inherit inputs from checks.
          checks = self.checks.${system};

          # Additional dev-shell environment variables can be set directly
          # MY_CUSTOM_DEVELOPMENT_VAR = "something else";

          shellHook = ''
            # For rust-analyzer 'hover' tooltips to work.
            export RUST_SRC_PATH="${toolchain}/lib/rustlib/src/rust/library";
          '';

          packages = with pkgs; [
            just
            xxd

            zellij # tmux alternative
            lurk # strace alternative
          ];
        };
      });
}
