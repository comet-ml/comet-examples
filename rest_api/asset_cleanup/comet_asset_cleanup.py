#!/usr/bin/env python3
"""
Comet ML Asset Cleanup Script

This script iterates through all experiments in a workspace/project, finds assets older than
a specified time threshold (default: 1 year), and deletes them to help manage cloud storage costs.

SAFETY: This script runs in DRY-RUN mode by default. Use --execute to actually delete assets.

Required Environment Variables:
- COMET_API_KEY: Your Comet ML API key
- COMET_WORKSPACE: Your workspace name (optional, can be passed as argument)
- COMET_PROJECT: Your project name (optional, can be passed as argument)

Usage:
    # Test what would be deleted (safe, default behavior)
    python asset_cleanup.py --workspace myworkspace --project myproject --days-threshold 365

    # Explicitly run in dry-run mode
    python asset_cleanup.py --workspace myworkspace --project myproject --dry-run

    # Actually delete assets (requires --execute flag)
    python asset_cleanup.py --workspace myworkspace --project myproject --days-threshold 365 --execute

The script uses the following Comet ML REST API endpoints:
- GET /api/rest/v2/experiments - Get all experiments in a project
- GET /api/rest/v2/experiment/asset/list - Get assets for each experiment
- GET /api/rest/v2/write/experiment/asset/delete - Delete individual assets
"""

import requests
import json
import os
import argparse
import time
from datetime import datetime, timedelta
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("comet_asset_cleanup.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class CometAssetCleanup:
    """
    Main class to handle Comet ML asset cleanup operations using REST API
    """

    def __init__(self, api_key: str, base_url: str = "https://www.comet.com"):
        """
        Initialize the cleanup client

        Args:
            api_key: Comet ML API key for authentication
            base_url: Base URL for Comet ML API (default: https://www.comet.com)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key, "Content-Type": "application/json"})

    def get_experiments(self, workspace_name: str, project_name: str, archived: bool = False) -> List[Dict]:
        """
        Get all experiments for a given project using GET /api/rest/v2/experiments

        Args:
            workspace_name: Name of the workspace
            project_name: Name of the project
            archived: Whether to include archived experiments (default: False)

        Returns:
            List of experiment dictionaries with keys:
            - experimentKey: Unique experiment identifier
            - experimentName: Name of the experiment
            - startTimeMillis: When experiment started (long timestamp)
            - endTimeMillis: When experiment ended (long timestamp)
            - durationMillis: How long experiment ran
        """
        url = f"{self.base_url}/api/rest/v2/experiments"
        params = {"workspaceName": workspace_name, "projectName": project_name, "archived": str(archived).lower()}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            experiments = data.get("experiments", [])
            logger.info(f"Retrieved {len(experiments)} experiments from {workspace_name}/{project_name}")
            return experiments

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get experiments: {e}")
            return []

    def get_projects(self, workspace_name: str) -> List[Dict]:
        """
        Get all projects for a given workspace using GET /api/rest/v2/projects

        Args:
            workspace_name: Name of the workspace

        Returns:
            List of project dictionaries. Expected keys include:
            - projectName: The project's name (preferred for downstream calls)
            - name: Sometimes provided instead of projectName
            - projectId: Project identifier (fallback if name missing)
        """
        url = f"{self.base_url}/api/rest/v2/projects"
        params = {"workspaceName": workspace_name}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            projects = data.get("projects", [])
            logger.info(f"Retrieved {len(projects)} projects from workspace {workspace_name}")
            return projects

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get projects for workspace {workspace_name}: {e}")
            return []

    def get_experiment_assets(self, experiment_key: str, asset_type: str = "all") -> List[Dict]:
        """
        Get all assets for a specific experiment using GET /api/rest/v2/experiment/asset/list

        Args:
            experiment_key: Unique experiment identifier
            asset_type: Type of assets to retrieve (default: "all")
                       Options: all, unknown, audio, video, image, histogram3d, etc.

        Returns:
            List of asset dictionaries with keys:
            - assetId: Unique asset identifier
            - fileName: Name of the asset file
            - createdAt: Timestamp when asset was created (long)
            - fileSize: Size of the file in bytes (long)
            - type: Type of the asset
            - step: Step when asset was logged (integer)
            - metadata: Metadata associated with the asset
            - runContext: Context when asset was logged
        """
        url = f"{self.base_url}/api/rest/v2/experiment/asset/list"
        params = {"experimentKey": experiment_key, "type": asset_type}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            assets = data.get("assets", [])
            logger.debug(f"Retrieved {len(assets)} assets for experiment {experiment_key}")
            return assets

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get assets for experiment {experiment_key}: {e}")
            return []

    def delete_asset(self, experiment_key: str, asset_id: str) -> bool:
        """
        Delete a specific asset using GET /api/rest/v2/write/experiment/asset/delete

        Args:
            experiment_key: Unique experiment identifier
            asset_id: Unique asset identifier to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        url = f"{self.base_url}/api/rest/v2/write/experiment/asset/delete"
        params = {"experimentKey": experiment_key, "assetId": asset_id}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            logger.info(f"Successfully deleted asset {asset_id} from experiment {experiment_key}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to delete asset {asset_id} from experiment {experiment_key}: {e}")
            return False

    def is_asset_old(self, asset: Dict, days_threshold: int) -> bool:
        """
        Check if an asset is older than the specified threshold

        Args:
            asset: Asset dictionary containing createdAt timestamp
            days_threshold: Number of days to consider as threshold

        Returns:
            True if asset is older than threshold, False otherwise
        """
        try:
            created_at = asset.get("createdAt")
            if not created_at:
                logger.warning(f"Asset {asset.get('assetId', 'unknown')} has no createdAt timestamp")
                return False

            # Convert milliseconds to seconds for datetime
            created_datetime = datetime.fromtimestamp(created_at / 1000.0)
            threshold_datetime = datetime.now() - timedelta(days=days_threshold)

            return created_datetime < threshold_datetime

        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing timestamp for asset {asset.get('assetId', 'unknown')}: {e}")
            return False

    def cleanup_old_assets(
        self,
        workspace_name: str,
        project_name: str,
        days_threshold: int = 365,
        dry_run: bool = True,
        batch_size: int = 10,
        delay_between_batches: float = 1.0,
        include_archived: bool = False,
        assume_yes: bool = False,
    ) -> Dict[str, int]:
        """
        Main cleanup function to delete old assets

        Args:
            workspace_name: Name of the workspace
            project_name: Name of the project
            days_threshold: Assets older than this many days will be deleted
            dry_run: If True, only report what would be deleted without actually deleting
            batch_size: Number of assets to delete in each batch
            delay_between_batches: Seconds to wait between batches to avoid rate limiting
            include_archived: If True, include archived experiments
            assume_yes: If True, skip interactive confirmation prompt (non-interactive)

        Returns:
            Dictionary with cleanup statistics:
            - total_experiments: Number of experiments processed
            - total_assets_found: Total number of assets found
            - assets_to_delete: Number of assets that match deletion criteria
            - assets_deleted: Number of assets successfully deleted
            - errors: Number of deletion errors encountered
        """
        stats = {
            "total_experiments": 0,
            "total_assets_found": 0,
            "assets_to_delete": 0,
            "assets_deleted": 0,
            "errors": 0,
            "total_size_freed": 0,
        }

        # Print prominent dry-run warning
        if dry_run:
            print("\n" + "=" * 80)
            print("🔍 DRY RUN MODE - NO ASSETS WILL BE DELETED")
            print("This is a simulation to show what would be deleted.")
            print("Use --execute flag to actually delete assets.")
            print("=" * 80)
        else:
            print("\n" + "⚠️ " * 10)
            print("🚨 LIVE DELETION MODE - ASSETS WILL BE PERMANENTLY DELETED!")
            print("⚠️ " * 10)

        logger.info(f"Starting asset cleanup for {workspace_name}/{project_name}")
        logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE DELETION'}")
        logger.info(f"Threshold: {days_threshold} days")
        logger.info(f"Include archived experiments: {include_archived}")

        # Get all experiments
        experiments = self.get_experiments(workspace_name, project_name, archived=include_archived)
        stats["total_experiments"] = len(experiments)

        if not experiments:
            logger.warning("No experiments found")
            return stats

        assets_to_process = []

        # Process each experiment
        for experiment in experiments:
            experiment_key = experiment["experimentKey"]
            experiment_name = experiment.get("experimentName", "Unnamed")

            logger.info(f"Processing experiment: {experiment_name} ({experiment_key})")

            # Get assets for this experiment
            assets = self.get_experiment_assets(experiment_key)
            stats["total_assets_found"] += len(assets)

            # Check each asset's age
            for asset in assets:
                if self.is_asset_old(asset, days_threshold):
                    file_size = asset.get("fileSize", 0)
                    created_at = asset.get("createdAt")
                    created_date = datetime.fromtimestamp(created_at / 1000.0) if created_at else None

                    asset_info = {
                        "experiment_key": experiment_key,
                        "experiment_name": experiment_name,
                        "asset_id": asset["assetId"],
                        "file_name": asset.get("fileName", "unknown"),
                        "file_size": file_size,
                        "created_at": created_at,
                        "created_date": created_date,
                        "asset_type": asset.get("type", "unknown"),
                        "age_days": (datetime.now() - created_date).days if created_date else None,
                    }
                    assets_to_process.append(asset_info)
                    stats["assets_to_delete"] += 1
                    stats["total_size_freed"] += file_size

        logger.info(f"Found {stats['assets_to_delete']} assets older than {days_threshold} days")
        logger.info(f"Total size to be freed: {self._format_bytes(stats['total_size_freed'])}")

        if dry_run:
            logger.info("🔍 DRY RUN - No assets will be deleted")
            # Build and save deletion plan before any deletions
            plan_path = self._save_deletion_plan(
                workspace_name=workspace_name, project_name=project_name, assets_to_process=assets_to_process
            )
            logger.info(f"Saved deletion plan to {plan_path}")
            self._print_assets_summary(assets_to_process, dry_run=True)
            return stats

        # Delete assets in batches (only in live mode)
        if assets_to_process:
            self._print_assets_summary(assets_to_process, dry_run=False)
            print(f"\n⚠️  You are about to PERMANENTLY DELETE {len(assets_to_process)} assets!")
            print(f"📊 Total storage to be freed: {self._format_bytes(stats['total_size_freed'])}")
            # Always save the deletion plan prior to asking for confirmation
            plan_path = self._save_deletion_plan(
                workspace_name=workspace_name, project_name=project_name, assets_to_process=assets_to_process
            )
            print(f"📝 Deletion plan saved to: {plan_path}")
            if not assume_yes:
                confirmation = input("\nType 'DELETE' (in capital letters) to confirm: ")
                if confirmation != "DELETE":
                    logger.info("❌ Deletion cancelled by user")
                    return stats
            else:
                logger.info("Proceeding without confirmation due to --yes flag")

        for i in range(0, len(assets_to_process), batch_size):
            batch = assets_to_process[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(assets_to_process) + batch_size - 1) // batch_size}")

            for asset_info in batch:
                try:
                    if self.delete_asset(asset_info["experiment_key"], asset_info["asset_id"]):
                        stats["assets_deleted"] += 1
                    else:
                        stats["errors"] += 1
                except Exception as e:
                    logger.error(f"Unexpected error deleting asset {asset_info['asset_id']}: {e}")
                    stats["errors"] += 1

            # Rate limiting delay
            if i + batch_size < len(assets_to_process):
                time.sleep(delay_between_batches)

        logger.info(f"✅ Cleanup completed. Deleted {stats['assets_deleted']} assets with {stats['errors']} errors")
        return stats

    def _save_deletion_plan(self, workspace_name: str, project_name: str, assets_to_process: List[Dict]) -> str:
        """Build a nested dict of assets to delete and save as JSON.

        Structure:
        {
          "<workspace>": {
            "<project>": {
              "<experiment_key>": ["<fileName>", ...]
            }
          }
        }
        """
        plan: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        if workspace_name not in plan:
            plan[workspace_name] = {}
        if project_name not in plan[workspace_name]:
            plan[workspace_name][project_name] = {}

        for asset in assets_to_process:
            experiment_key = asset["experiment_key"]
            file_name = asset.get("file_name", "unknown")
            if experiment_key not in plan[workspace_name][project_name]:
                plan[workspace_name][project_name][experiment_key] = []
            plan[workspace_name][project_name][experiment_key].append(file_name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"asset_deletion_plan_{workspace_name}_{project_name}_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        return filename

    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes into human readable format"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_count < 1024.0:
                return f"{bytes_count:.2f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.2f} PB"

    def _print_assets_summary(self, assets: List[Dict], dry_run: bool = True) -> None:
        """Print a summary of assets that would be deleted"""
        if not assets:
            return

        mode_indicator = "🔍 DRY RUN - ASSETS TO BE DELETED" if dry_run else "🚨 LIVE MODE - ASSETS TO DELETE"

        print("\n" + "=" * 80)
        print(mode_indicator)
        print("=" * 80)

        by_experiment = {}
        for asset in assets:
            exp_name = asset["experiment_name"]
            if exp_name not in by_experiment:
                by_experiment[exp_name] = []
            by_experiment[exp_name].append(asset)

        for exp_name, exp_assets in by_experiment.items():
            total_size = sum(a["file_size"] for a in exp_assets)
            avg_age = sum(a.get("age_days", 0) for a in exp_assets if a.get("age_days")) / len(exp_assets)

            print(f"\n📁 Experiment: {exp_name}")
            print(f"   Assets to {'simulate deletion' if dry_run else 'DELETE'}: {len(exp_assets)}")
            print(f"   Total size: {self._format_bytes(total_size)}")
            print(f"   Average age: {avg_age:.0f} days")

            if len(exp_assets) <= 5:
                for asset in exp_assets:
                    age_str = f"({asset.get('age_days', '?')} days old)" if asset.get("age_days") else ""
                    print(f"     - {asset['file_name']} ({asset['asset_type']}) - {self._format_bytes(asset['file_size'])} {age_str}")
            else:
                for asset in exp_assets[:3]:
                    age_str = f"({asset.get('age_days', '?')} days old)" if asset.get("age_days") else ""
                    print(f"     - {asset['file_name']} ({asset['asset_type']}) - {self._format_bytes(asset['file_size'])} {age_str}")
                print(f"     ... and {len(exp_assets) - 3} more assets")

        print("=" * 80)


def main():
    """Main function to handle command line arguments and run cleanup"""
    parser = argparse.ArgumentParser(
        description="Clean up old Comet ML assets to reduce storage costs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Safe dry-run mode (default) - shows what would be deleted
    python %(prog)s --workspace myws --project myproj --days-threshold 365

    # Explicit dry-run mode
    python %(prog)s --workspace myws --project myproj --dry-run

    # Actually delete assets (requires explicit --execute flag)
    python %(prog)s --workspace myws --project myproj --days-threshold 365 --execute
        """,
    )

    parser.add_argument("--workspace", required=False, help="Workspace name (can also use COMET_WORKSPACE env var)")

    parser.add_argument("--project", required=False, help="Project name (can also use COMET_PROJECT env var)")

    parser.add_argument("--days-threshold", type=int, default=365, help="Delete assets older than this many days (default: 365)")

    # Enhanced dry-run argument with clearer messaging
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="🔍 Run in simulation mode - show what would be deleted without actually deleting (DEFAULT behavior for safety)",
    )

    parser.add_argument("--execute", action="store_true", help="🚨 ACTUALLY DELETE assets - overrides dry-run mode. PERMANENT ACTION!")

    parser.add_argument("--batch-size", type=int, default=10, help="Number of assets to delete in each batch (default: 10)")

    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between batches (default: 1.0)")

    parser.add_argument("--api-key", help="Comet ML API key (can also use COMET_API_KEY env var)")

    parser.add_argument("--base-url", default="https://www.comet.com", help="Base URL for Comet ML API (default: https://www.comet.com)")

    parser.add_argument("--include-archived", action="store_true", help="Include archived experiments in cleanup (default: False)")

    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation prompts (non-interactive runs)")

    args = parser.parse_args()

    # Get configuration from args or environment
    api_key = args.api_key or os.getenv("COMET_API_KEY")
    workspace = args.workspace or os.getenv("COMET_WORKSPACE")
    project = args.project or os.getenv("COMET_PROJECT")

    if not api_key:
        print("❌ Error: API key is required.")
        print("Set COMET_API_KEY environment variable or use --api-key")
        return 1

    if not workspace:
        print("❌ Error: Workspace is required.")
        print("Set COMET_WORKSPACE environment variable or use --workspace")
        return 1

    # Project is optional: if omitted, iterate across all projects in workspace

    # Create cleanup client
    cleanup = CometAssetCleanup(api_key, args.base_url)

    # Determine if this is a dry run - defaults to True for safety
    dry_run = not args.execute

    if dry_run:
        print("🔍 Running in DRY-RUN mode (safe, default behavior)")
        print("   Use --execute flag to actually delete assets")
    else:
        print("🚨 Running in EXECUTE mode - will actually delete assets!")

    # Run cleanup for one or many projects
    try:
        all_stats: List[Dict[str, int]] = []
        projects_to_process: List[str] = []
        if project:
            projects_to_process = [project]
        else:
            projects_meta = cleanup.get_projects(workspace)
            if not projects_meta:
                print("❌ Error: No projects found in workspace.")
                return 1
            # Normalize project name across possible keys
            for p in projects_meta:
                p_name = p.get("projectName") or p.get("name")
                if p_name:
                    projects_to_process.append(p_name)
            if not projects_to_process:
                print("❌ Error: Unable to resolve project names from API response.")
                return 1

        for proj_name in projects_to_process:
            logger.info(f"\n==== Processing project: {workspace}/{proj_name} ====")
            stats = cleanup.cleanup_old_assets(
                workspace_name=workspace,
                project_name=proj_name,
                days_threshold=args.days_threshold,
                dry_run=dry_run,
                batch_size=args.batch_size,
                delay_between_batches=args.delay,
                include_archived=args.include_archived,
                assume_yes=args.yes,
            )
            all_stats.append(stats)

        # Print final statistics
        print("\n" + "=" * 60)
        print("📊 CLEANUP STATISTICS")
        print("=" * 60)
        print(f"Mode: {'🔍 DRY RUN (simulation)' if dry_run else '🚨 LIVE EXECUTION'}")
        # Aggregate stats across projects
        agg = {
            "total_experiments": sum(s["total_experiments"] for s in all_stats),
            "total_assets_found": sum(s["total_assets_found"] for s in all_stats),
            "assets_to_delete": sum(s["assets_to_delete"] for s in all_stats),
            "assets_deleted": sum(s["assets_deleted"] for s in all_stats),
            "errors": sum(s["errors"] for s in all_stats),
            "total_size_freed": sum(s["total_size_freed"] for s in all_stats),
        }

        print(f"Experiments processed: {agg['total_experiments']}")
        print(f"Total assets found: {agg['total_assets_found']}")
        print(f"Assets meeting deletion criteria: {agg['assets_to_delete']}")
        if not dry_run:
            print(f"Assets successfully deleted: {agg['assets_deleted']}")
            print(f"Deletion errors: {agg['errors']}")
        else:
            print(f"Assets that would be deleted: {agg['assets_to_delete']}")
        print(f"Storage space {'that would be freed' if dry_run else 'freed'}: {cleanup._format_bytes(agg['total_size_freed'])}")

        if dry_run and stats["assets_to_delete"] > 0:
            print("\n🔍 This was a dry run. To actually delete these assets, run with --execute")

        print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
