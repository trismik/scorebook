"""Authentication CLI commands."""

import argparse
import getpass
import sys

from scorebook.trismik.credentials import get_stored_token, get_token_path, login, logout, whoami


def auth_command(args: argparse.Namespace) -> int:
    """Handle auth subcommands."""
    if args.auth_command == "login":
        return login_command(args)
    elif args.auth_command == "logout":
        return logout_command(args)
    elif args.auth_command == "whoami":
        return whoami_command(args)
    else:
        print(
            "Error: No auth command specified. Use 'login', 'logout', or 'whoami'.", file=sys.stderr
        )
        return 1


def login_command(args: argparse.Namespace) -> int:
    """Handle login command."""
    try:
        token = args.token

        if not token:
            # Check if we're already logged in
            stored_token = get_stored_token()

            if stored_token:
                print("You are already logged in.")
                overwrite = (
                    input("Do you want to overwrite the existing token? (y/N): ").lower().strip()
                )
                if overwrite not in ("y", "yes"):
                    print("Login cancelled.")
                    return 0

            # Prompt for token securely
            print("Enter your Trismik API token:")
            print("(You can find your token at: https://trismik.com/settings/tokens)")
            token = getpass.getpass("Token: ").strip()

        if not token:
            print("Error: No token provided.", file=sys.stderr)
            return 1

        # Login
        login(token)

        # Success message
        print(f"Successfully logged in! Token saved to {get_token_path()}")
        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def logout_command(args: argparse.Namespace) -> int:
    """Handle logout command."""
    try:
        success = logout()
        if success:
            print("Successfully logged out!")
        else:
            print("Not currently logged in.")
        return 0
    except Exception as e:
        print(f"Error during logout: {e}", file=sys.stderr)
        return 1


def whoami_command(args: argparse.Namespace) -> int:
    """Handle whoami command."""
    try:
        token = whoami()
        if token is None:
            print("Not logged in. Run 'scorebook auth login' first.")
            return 1
        else:
            # TODO: Make actual API call to get user info
            # For now, just confirm we have a token
            print(f"Logged in with token: {token[:8]}...")
            return 0
    except Exception as e:
        print(f"Error checking login status: {e}", file=sys.stderr)
        return 1
