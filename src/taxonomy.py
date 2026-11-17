"""
taxonomy.py
===========
Authoritative IT Ticket Classification Taxonomy for HCLTech Hackathon Phase 2.

Grounds all classification targets in the verified historical dataset (400 tickets)
spanning 6 enterprise functional categories.
"""

TAXONOMY_VERSION = "v1.0"

TAXONOMY = {
    "Network": {
        "definition": "Issues concerning data transmission, IP connectivity, VPN tunnels, Wi-Fi access points, firewall rule filtering, and DNS name resolution.",
        "subcategories": ["VPN", "DNS", "Wi-Fi", "Firewall", "Bandwidth", "Switch/Router"],
        "representative_examples": [
            "VPN disconnects with error 800 after authentication",
            "GlobalProtect VPN connection drops repeatedly every 5 minutes",
            "Cannot resolve internal hostnames via corp DNS server 10.10.1.2",
            "Office 4th floor Wi-Fi signal degraded and dropping packets",
            "Firewall blocking outbound HTTPS traffic to AWS S3 bucket"
        ],
        "common_ambiguities": [
            "Confusion with 'Access & Identity': VPN authentication failures (MFA vs network gateway timeout).",
            "Confusion with 'Server & Infrastructure': API gateway 502 error caused by network connectivity vs backend server OOM crash."
        ],
        "min_training_examples": 50,
        "actual_dataset_count": 74
    },
    "Access & Identity": {
        "definition": "User authentication, single sign-on (SSO), Active Directory domain accounts, multi-factor authentication (MFA), password expirations, and role-based permissions.",
        "subcategories": ["Account Lockout", "Password Reset", "Permissions / Access Request", "MFA / 2FA", "Active Directory"],
        "representative_examples": [
            "User account locked out after multiple failed logon attempts (Event ID 4625)",
            "Account locked out repeatedly every 15 minutes via mobile ActiveSync",
            "Urgent self-service password reset token expired",
            "MFA push notification not arriving on Microsoft Authenticator app",
            "Access request for Finance shared folder 'N:\\Finance\\FY26_Budgets'"
        ],
        "common_ambiguities": [
            "Confusion with 'Email & Collaboration': Inability to access shared mailbox due to permission denied vs Exchange server synchronization delay.",
            "Confusion with 'Network': MFA prompt failing due to network timeout vs expired credentials."
        ],
        "min_training_examples": 50,
        "actual_dataset_count": 57
    },
    "Hardware": {
        "definition": "Physical client endpoint hardware failures, battery swelling, docking stations, displays, shared network printers, and peripheral devices.",
        "subcategories": ["Desktop", "Docking Station", "Printer", "Laptop", "Monitor", "Peripheral"],
        "representative_examples": [
            "Dell Latitude 5430 battery swelling and keyboard bulging",
            "Network printer PRN-FL3-COLOR in error state with paper jam 13.00.00",
            "Dual monitor display flickering via Thunderbolt docking station WD19TBS",
            "Blue Screen of Death (BSOD) DRIVER_IRQL_NOT_LESS_OR_EQUAL on boot (nvlddmkm.sys)"
        ],
        "common_ambiguities": [
            "Confusion with 'Software': BSOD caused by hardware GPU failure vs corrupted software graphics driver.",
            "Confusion with 'Network': Network printer unreachable due to paper jam vs offline IP lease."
        ],
        "min_training_examples": 50,
        "actual_dataset_count": 69
    },
    "Software": {
        "definition": "Operating system configuration, productivity applications (Office 365, browsers), enterprise developer IDEs, and local software deployment packages.",
        "subcategories": ["Developer Tools", "Browser", "Operating System", "Office 365", "VPN Client", "Antivirus"],
        "representative_examples": [
            "Microsoft Outlook crashing immediately upon opening profile (OUTLMAPI.DLL)",
            "Software installation error 0x80070005 Access is denied during CAD install",
            "Google Chrome showing STATUS_ACCESS_VIOLATION on intranet portals",
            "Windows Update failure error 0x80248007 on Windows 11 Enterprise"
        ],
        "common_ambiguities": [
            "Confusion with 'Email & Collaboration': Outlook application crash vs Exchange mailbox quota limit.",
            "Confusion with 'Access & Identity': Software installation blocked by admin privileges vs antivirus quarantine."
        ],
        "min_training_examples": 50,
        "actual_dataset_count": 68
    },
    "Server & Infrastructure": {
        "definition": "Core backend application servers, database instances, virtualization hosts (VMware vSphere), enterprise SAN storage, and cloud pipeline services.",
        "subcategories": ["Web Server", "Storage & Backup", "Database", "Virtual Machine", "Cloud Service"],
        "representative_examples": [
            "Production API gateway returning HTTP 502 Bad Gateway to clients",
            "Database connection pool exhausted on prod-db-01.corp (max_connections=200)",
            "ESXi host host-04.datacenter lost storage path to SAN lun_vmfs_02",
            "Backup job failed: VSS writer error code 0x800423f4 on FileServer-01"
        ],
        "common_ambiguities": [
            "Confusion with 'Network': Database unreachable due to port blocking vs database listener crashed.",
            "Confusion with 'Software': Web server crash due to application code bug vs server memory exhaustion."
        ],
        "min_training_examples": 50,
        "actual_dataset_count": 61
    },
    "Email & Collaboration": {
        "subcategories": ["Shared Mailbox", "Calendar / Scheduling", "Microsoft Teams", "Outlook / Exchange"],
        "definition": "Exchange Online email routing, shared departmental inboxes, Microsoft Teams audio/video conferencing, and calendar room booking auto-attendants.",
        "representative_examples": [
            "Emails stuck in Outlook Outbox not sending to external recipients (0x8004210B)",
            "User unable to access Shared Mailbox 'procurement-inbox@corp.com'",
            "Microsoft Teams microphone and video feed glitching during conference calls",
            "Room booking calendar decline: Conflicting meeting room reservation"
        ],
        "common_ambiguities": [
            "Confusion with 'Software': Teams client lagging due to local CPU load vs Exchange server disruption.",
            "Confusion with 'Access & Identity': Shared mailbox permission missing vs Outlook profile corrupted."
        ],
        "min_training_examples": 50,
        "actual_dataset_count": 71
    }
}

CATEGORIES = list(TAXONOMY.keys())
