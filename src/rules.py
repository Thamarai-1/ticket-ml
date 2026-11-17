"""
rules.py
========
Documented Operational Rules for Subcategory Hierarchy Mapping and Triage Fields.

In strict compliance with Section 11:
1. Subcategories are mapped using deterministic, domain-verified keyword matching
   conditioned ONLY on the predicted primary category.
2. If no documented rule matches with high specificity, subcategory returns None.
3. Priority and intent are NOT fabricated or guessed from vague symptom text alone
   (they return None / needs-review), because genuine ITIL prioritization requires
   verified business impact and urgency inputs.
"""

from taxonomy import TAXONOMY

SUBCATEGORY_RULES = {
    "Network": [
        (["vpn", "globalprotect", "tunnel", "gateway 10."], "VPN"),
        (["dns", "nxdomain", "nslookup", "resolver", "hostname"], "DNS"),
        (["wi-fi", "wifi", "ssid", "wireless", "ap-", "signal", "channel"], "Wi-Fi"),
        (["firewall", "palo alto", "egress", "sec-rule", "port 443"], "Firewall"),
        (["bandwidth", "saturation", "latency", "packet loss"], "Bandwidth"),
        (["switch", "router", "vlan", "trunk", "cisco"], "Switch/Router")
    ],
    "Access & Identity": [
        (["locked out", "lockout", "event id 4625", "badpwdcount", "ad account"], "Account Lockout"),
        (["password reset", "sspr", "expired password", "totp", "one-time password"], "Password Reset"),
        (["mfa", "2fa", "authenticator", "push notification", "qr code"], "MFA / 2FA"),
        (["permission", "access request", "shared folder", "ntfs", "security group", "sg-"], "Permissions / Access Request"),
        (["active directory", "domain controller", "pdc", "ad schema"], "Active Directory")
    ],
    "Hardware": [
        (["battery swelling", "bulging", "laptop chassis", "trackpad", "dell latitude"], "Laptop"),
        (["paper jam", "printer", "spooler", "print queue", "13.00.00", "toner"], "Printer"),
        (["docking station", "thunderbolt", "wd19", "dual monitor", "flickering"], "Docking Station"),
        (["bsod", "blue screen", "stop code", "nvlddmkm", "memtest", "desktop"], "Desktop"),
        (["monitor", "displayport", "hdmi", "screen"], "Monitor")
    ],
    "Software": [
        (["outlook", "outlmapi.dll", "ost", "pst", "add-in", "safe mode"], "Office 365"),
        (["0x80070005", "autocad", "intune", "company portal", "installer"], "Developer Tools"),
        (["chrome", "browser", "status_access_violation", "webgui", "edge"], "Browser"),
        (["windows update", "0x80248007", "kb5034441", "softwaredistribution", "dism", "sfc"], "Operating System"),
        (["antivirus", "crowdstrike", "falcon", "quarantine"], "Antivirus")
    ],
    "Server & Infrastructure": [
        (["502 bad gateway", "nginx", "reverse proxy", "upstream", "web-gw"], "Web Server"),
        (["database", "postgres", "connection pool", "max_connections", "pgbouncer"], "Database"),
        (["esxi", "vsphere", "vmfs", "storage vmotion", "sfp+", "fiber channel"], "Virtual Machine"),
        (["backup", "vss", "0x800423f4", "veeam", "purestorage"], "Storage & Backup"),
        (["kubernetes", "k8s", "pod", "cluster", "aws", "cloud"], "Cloud Service")
    ],
    "Email & Collaboration": [
        (["outbox", "0x8004210b", "attachment", "large file", "sending"], "Outlook / Exchange"),
        (["shared mailbox", "procurement-inbox", "add-mailboxpermission", "automapping"], "Shared Mailbox"),
        (["teams", "microphone", "video feed", "glitching", "conference call", "call quality"], "Microsoft Teams"),
        (["room booking", "calendar", "boardroom", "meeting invite", "orphaned"], "Calendar / Scheduling")
    ]
}

def resolve_subcategory(predicted_category: str, text: str) -> str:
    """
    Applies deterministic hierarchy rules to infer subcategory
    strictly within the scope of the predicted category.
    """
    if not predicted_category or predicted_category not in SUBCATEGORY_RULES:
        return None

    text_lower = text.lower()
    rules = SUBCATEGORY_RULES[predicted_category]

    for keywords, subcat in rules:
        if any(kw in text_lower for kw in keywords):
            # Verify subcategory is valid for this category
            valid_subcats = TAXONOMY.get(predicted_category, {}).get("subcategories", [])
            if subcat in valid_subcats:
                return subcat

    return None

def resolve_priority_and_intent(text: str) -> tuple[str, str]:
    """
    Per Section 11, priority and intent cannot be responsibly determined
    from ungrounded symptom text alone. Returns (None, None) so the API
    honestly states that downstream triage is required.
    """
    return (None, None)
