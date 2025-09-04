"""Main CLI entry point for scorebook."""

import argparse
import sys
from typing import List, Optional

from .auth import auth_command


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="scorebook",
        description="Scorebook CLI - A Python project for LLM evaluation",
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Auth subcommand
    auth_parser = subparsers.add_parser("auth", help="Authentication commands")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command", help="Auth commands")

    # Auth login
    login_parser = auth_subparsers.add_parser("login", help="Login to scorebook")
    login_parser.add_argument("--token", help="API token to use for login")

    # Auth logout
    auth_subparsers.add_parser("logout", help="Logout from scorebook")

    # Auth whoami
    auth_subparsers.add_parser("whoami", help="Show current login status")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "auth":
            return auth_command(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
